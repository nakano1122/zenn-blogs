---
title: "pytorch-lightningで検索モデルをシンプルに実装する方法"
emoji: "🔎"
type: "tech"
topics: ["pytorch", "lightning", "machinelearning", "情報検索", "python"]
published: true
---

:::details この記事で分かること

- `LightningDataModule` で query と document を batch へ組み立てる流れ
- `compute_loss` だけを検索モデル側に実装する学習処理
- `Trainer.predict` と逐次 writer を組み合わせたランキング結果の保存

:::

検索モデルの実験では、score を計算する部分よりも、その周辺のコードが先に複雑になります。学習では query と document を読み込み、loss と validation を計算します。ランキングでは、学習済みモデルで候補文書をスコアリングし、query や document のIDと一緒に保存します。

この処理を学習用とランキング用に別々に書くと、encoder や score 計算が重複します。

例えば、次のように学習用とランキング用の関数へ同じ処理が現れます。

```python
def train_step(batch):
    query_embeddings = encode_queries(batch["queries"])
    document_embeddings = encode_documents(batch["positive_documents"])
    scores = compute_scores(query_embeddings, document_embeddings)
    return compute_loss(scores, batch["labels"])


def rank(batch):
    query_embeddings = encode_queries(batch["queries"])
    document_embeddings = encode_documents(batch["candidate_documents"])
    scores = compute_scores(query_embeddings, document_embeddings)
    return save_scores(scores, batch["metadata"])
```

この重複を避けるために、score 計算をモデルへ集め、学習とランキングの差分だけをそれぞれの処理へ残します。以降では、PyTorch Lightning を使ってこの境界を実装します。

## 共通する score 計算をモデルに集める

最初に、学習とランキングの両方から呼び出す処理を決めます。ここでは、query の文字列を query embedding へ変換し、document の文字列を document embedding へ変換して、両者の score 行列を返すモデルにします。

```python
from collections.abc import Sequence

import lightning as L
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts: Sequence[str]) -> Tensor:
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        outputs = self.model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        return (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(
            dim=1
        ).clamp_min(1)


class BiEncoderModel(L.LightningModule):
    def __init__(
        self,
        query_encoder: TextEncoder,
        document_encoder: TextEncoder,
    ) -> None:
        super().__init__()
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder

    def encode_queries(self, queries: Sequence[str]) -> Tensor:
        embeddings = self.query_encoder.encode(queries)
        return F.normalize(embeddings, dim=-1)

    def encode_documents(self, documents: Sequence[str]) -> Tensor:
        embeddings = self.document_encoder.encode(documents)
        return F.normalize(embeddings, dim=-1)

    def forward(
        self,
        queries: Sequence[str],
        documents: Sequence[str],
    ) -> Tensor:
        query_embeddings = self.encode_queries(queries)
        document_embeddings = self.encode_documents(documents)
        return query_embeddings @ document_embeddings.T
```

`forward` は、query と document の全組み合わせを score 行列として返します。`scores[i, j]` は、`queries[i]` と `documents[j]` の組み合わせに対応します。

このインターフェースを先に決めると、学習側は score から loss を作るだけになります。ランキング側は候補文書を渡して score を受け取り、保存すれば済みます。次は、この入力をどこで作るかを切り出します。

`TextEncoder` は Transformers の事前学習済みモデルを使って embedding を作ります。query 用と document 用にそれぞれインスタンスを作り、`BiEncoderModel` へ渡します。モデル名や checkpoint の保存場所を `BiEncoderModel` が解決しないため、encoder を差し替えても score 計算のコードは変わりません。

`AutoModel.from_pretrained` と `AutoTokenizer.from_pretrained` は、指定したモデル名やパスに対応する事前学習済みモデルと tokenizer を読み込む API です。

@[card](https://huggingface.co/docs/transformers/en/models)

## DataModuleでモデルへの入力を作る

モデルが文字列の query と document を受け取るなら、データセットの列名や document の保存形式はモデルから隠せます。その役割を `LightningDataModule` と文書リポジトリに任せます。

```python
import lightning as L
from torch.utils.data import DataLoader


class DocumentRepository:
    def __init__(self, document_path: str) -> None:
        self.document_path = document_path

    def get_texts(self, document_ids: list[str]) -> list[str]:
        # document_ids に必要な文書だけを読み込む。
        raise NotImplementedError


class SearchDataModule(L.LightningDataModule):
    def __init__(self, dataset, document_repository: DocumentRepository) -> None:
        super().__init__()
        self.dataset = dataset
        self.document_repository = document_repository

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=self.collate_train,
        )

    def collate_train(self, records: list[dict]) -> dict[str, list[str]]:
        queries = []
        document_ids = []
        for record in records:
            queries.append(record["query"])
            document_ids.append(record["positive_document_id"])

        return {
            "queries": queries,
            "positive_documents": self.document_repository.get_texts(document_ids),
        }
```

`collate_train` はデータセットの record を、モデルが受け取る batch へ変換します。document の本文は batch に含まれる ID だけを `DocumentRepository` から取得します。そのため、全 document を最初からメモリへ展開せずに学習できます。

この分け方なら、DataModule がデータセットの列名と文書リポジトリを受け持ち、モデルは文字列の query と document だけを扱います。Lightning の DataModule は、データの準備と DataLoader をまとめて管理するための仕組みです。

@[card](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)

### 長い文書の扱いをここで決める

文書が encoder の入力上限を超えると、DataModule から渡す前後で追加の処理が必要になります。代表的な選択肢は次の3つです。

- 上限を超えた部分を切り捨てる
- 文書を window に分割し、window ごとに score を計算する
- window の embedding を pooling してから document score を計算する

この記事の構成では、文書を window に分割し、window ごとの score を document 単位へ集約します。window の最大 score を採用すると、文書の一部分が query と強く一致したときに、その文書を候補へ残せます。

一方、文書全体の情報を平均的に評価したいタスクでは、mean pooling など別の集約方法が向く可能性があります。ここはモデルの都合だけで決めず、検索対象の文書と relevance の定義に合わせて選びます。

入力の形が決まったので、学習側ではこの batch から loss を作ります。

## 学習側は loss の計算に絞る

Bi-Encoder の学習で検索モデル固有の処理になるのは、score から正解ラベルを使って loss を計算する部分です。学習 loop、device の移動、validation の呼び出し、checkpoint の管理は Lightning の `Trainer` に任せます。

```python
import torch
import torch.nn.functional as F
from torch import Tensor


class SearchTrainer(BiEncoderModel):
    def __init__(
        self,
        query_encoder: TextEncoder,
        document_encoder: TextEncoder,
        learning_rate: float = 2e-5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(query_encoder, document_encoder)
        self.learning_rate = learning_rate
        self.temperature = temperature

    def compute_loss(self, batch: dict[str, list[str]]) -> Tensor:
        queries = batch["queries"]
        positive_documents = batch["positive_documents"]
        scores = self(queries, positive_documents)

        # positive_documents の順序は queries と対応させておく。
        targets = torch.arange(scores.size(0), device=scores.device)
        return F.cross_entropy(scores / self.temperature, targets)

    def training_step(self, batch: dict[str, list[str]], batch_idx: int) -> Tensor:
        del batch_idx
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )
```

`compute_loss` ではモデルに batch を渡して score を作り、loss 関数の結果を返しています。ここで示した cross entropy は、Lightning 側へ検索モデル固有の loss を差し込む例です。loss の選択や負例の設計には複数の方法があるため、この記事では扱いません。重要なのは、loss を `compute_loss` に閉じ込めることで、`training_step` や optimizer の設定を共通化できる点です。

ここまでで学習に必要なモデルと batch がそろいました。ランキングでは同じ score 計算を使い、学習用の loss だけを外します。

## ランキング側は予測と保存を分ける

学習が終わると、query と候補 document を採点してランキング結果を作ります。この処理では、score の計算と結果ファイルの生成を分けます。学習済みモデルを読み込んだ後は、学習時と同じ embedding・score 計算を利用できます。

候補文書数が query ごとに異なる場合は、候補文書を一度平坦化し、各文書がどの query に属するかを index で管理します。

```python
import torch


def score_candidates(
    model: BiEncoderModel,
    queries: list[str],
    candidate_documents: list[str],
    candidate_counts: list[int],
) -> Tensor:
    query_embeddings = model.encode_queries(queries)
    document_embeddings = model.encode_documents(candidate_documents)
    query_indexes = torch.arange(
        len(queries),
        device=query_embeddings.device,
    ).repeat_interleave(
        torch.tensor(candidate_counts, device=query_embeddings.device),
    )

    return (
        query_embeddings[query_indexes] * document_embeddings
    ).sum(dim=-1)
```

この関数は、入力した `candidate_documents` と同じ順序で score を返します。候補文書との対応を崩さずに後段へ渡せるため、モデル計算と結果整形を分けられます。

Lightning では `predict_step` に予測処理を実装し、`Trainer.predict` で batch ごとに推論します。

```python
class RankingModule(BiEncoderModel):
    def predict_step(
        self,
        batch: dict[str, object],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        del batch_idx, dataloader_idx
        return score_candidates(
            self,
            queries=batch["queries"],
            candidate_documents=batch["candidate_documents"],
            candidate_counts=batch["candidate_counts"],
        )
```

@[card](https://lightning.ai/docs/pytorch/stable/common/trainer.html)

### 結果は batch 単位で書き出す

`Trainer.predict` の結果をすべてリストへ集めると、候補文書数に応じてホスト側のメモリを使います。ランキング対象が大きい場合は、予測 batch ごとに結果を書き出す方が扱いやすくなります。

```python
from lightning.pytorch.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_path: str) -> None:
        super().__init__(write_interval="batch")
        self.output_path = output_path
        self.temporary_path = f"{output_path}.tmp"
        self.writer = None

    def on_predict_start(self, trainer, pl_module) -> None:
        del trainer, pl_module
        self.writer = open_result_writer(self.temporary_path)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:
        del trainer, pl_module, batch_indices, batch_idx, dataloader_idx
        self.writer.write(batch["metadata"], prediction.detach().cpu())
```

writer は score と query ID、document ID、順位などを結果の1行へ変換します。書き込み先は最終ファイルではなく一時ファイルにし、`Trainer.predict` が成功した場合だけ最終パスへ置き換えます。

```python
writer = PredictionWriter("ranking.parquet")
trainer = L.Trainer(
    callbacks=[writer],
    enable_checkpointing=False,
)

try:
    trainer.predict(
        model,
        datamodule=ranking_data,
        return_predictions=False,
    )
except BaseException:
    writer.abort()
    raise
else:
    writer.close()
```

`open_result_writer` は一時ファイルへ結果を書き込む writer を生成します。`close` は writer を閉じて一時ファイルを最終パスへ置き換え、`abort` は一時ファイルを削除して既存の最終結果を残します。ここでは保存形式に依存しないよう、実装を省略しています。

この構成なら、予測結果を全件保持せずに済みます。途中で失敗しても、書きかけの結果で既存ファイルを上書きしません。`BasePredictionWriter` は batch 単位の保存を拡張するための callback です。

@[card](https://lightning.ai/docs/pytorch/stable/common/trainer.html)

@[card](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BasePredictionWriter.html)

## この分け方が効く場面

ここまでの処理を、次の責務へ分けました。

| 責務                  | 担当               | 主な入出力                     |
| --------------------- | ------------------ | ------------------------------ |
| query のベクトル化    | `QueryEncoder`     | query → query embedding        |
| document のベクトル化 | `DocumentEncoder`  | document → document embedding  |
| 関連度の計算          | `BiEncoderModel`   | embedding → score              |
| loss の計算           | `SearchTrainer`    | score → loss                   |
| batch の準備          | `SearchDataModule` | record → model input           |
| 推論結果の保存        | `PredictionWriter` | score + metadata → result file |

この分け方が効くのは、学習とランキングで同じ embedding・score 計算を使う場面です。encoder を差し替えるときも、モデル本体ではなく注入する部品と設定を変更します。保存形式を変えるときも、モデル側には影響しません。

ただし、分割すれば必ず実装が短くなるわけではありません。小規模な検証や候補数の少ない処理なら、単一のスクリプトの方が読みやすいこともあります。DataModule や writer を分ける価値が出るのは、学習・評価・ランキングを繰り返す場合や、データ量のためにメモリ管理が問題になる場合です。

document embedding の事前計算も、常に採用する必要はありません。事前計算すればランキング時の処理を減らせますが、文書更新時の再計算と index の整合性を管理する必要があります。検索対象が頻繁に変わる場合は、ランキング用 batch 内で document encoder を実行する方が運用しやすい場合があります。

## まとめ

検索モデルの実装で最初に決めるべきなのは、Lightning の設定ではなく、モデルと周辺処理の境界です。

query/document の embedding と score 計算を `BiEncoderModel` に集め、DataModule から必要な入力だけを渡します。学習側には loss を、ランキング側には prediction と保存を担当させると、同じ計算を使いながら役割を分けられます。

さらに、結果を batch 単位で書き出せば、ランキング対象をすべてメモリへ載せずに済みます。encoder や loss を試すときに変更箇所が広がらないことが、この構成の実用的な利点です。
