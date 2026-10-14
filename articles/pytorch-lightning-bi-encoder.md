---
title: "pytorch-lightningで検索モデルをシンプルに実装"
emoji: "🔎"
type: "tech"
topics: ["pytorch", "lightning", "machinelearning", "情報検索", "python"]
published: false
---

:::details この記事で分かること

- クエリエンコーダーと文書エンコーダーを分け、適合度の計算を共通化する構成
- `LightningDataModule` の collate function でクエリと文書を batch へ組み立てる流れ
- `compute_loss` に検索モデル固有の loss を集約する学習処理
- `Trainer.predict` と batch 単位の writer を組み合わせたランキング結果の保存

:::

検索モデルの実験コードには、モデル本体だけでなく、データセットからの batch 作成、学習、ランキング結果の保存が含まれます。学習とランキングは用途が異なりますが、どちらもクエリと文書を embedding に変換し、適合度を計算する処理を使います。

そのため、学習処理とランキング処理がそれぞれエンコーダーと適合度の計算を持つと、同じ処理が複数箇所に現れます。共通する計算をモデルへ集約し、学習とランキングに固有の処理だけを分けると、変更箇所を限定できます。

PyTorch Lightning では、共通する計算を `LightningModule` に置き、データ処理を `LightningDataModule` に、実行を `Trainer` に委ねられます。以下では、この分担に沿ってモデル、学習、ランキングの順に実装します。

コード例は、`lightning>=2.6,<2.7`、`torch>=2.2`、`transformers>=4.49,<5` を前提としています。

## `LightningModule` に embedding と適合度を集約する

最初に、クエリと文書を embedding（埋め込みベクトル）へ変換し、両者の適合度を計算する処理を定義します。次の `BiEncoderModel` は、この共通処理を `LightningModule` の `forward` にまとめています。

```python
from collections.abc import Sequence

import lightning as L
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, max_length: int) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

    def encode(self, texts: Sequence[str]) -> Tensor:
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
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

`forward` は、クエリと文書の全ペアに対する適合度行列を返します。`scores[i, j]` は、`queries[i]` と `documents[j]` のペアに対応します。embedding を L2 正規化してから内積を取るため、ここでの適合度は cosine 類似度です。

このクラスの責務は、テキストから embedding と適合度を計算することだけです。したがって、学習モジュールは適合度から loss を計算し、ランキングモジュールは候補文書ごとの適合度をランキング結果へ変換できます。

`TextEncoder` は Transformers の事前学習済みモデルで embedding を生成します。クエリ用と文書用のエンコーダーは、`BiEncoderModel` の外で生成して渡します。モデル名や checkpoint の保存場所を `BiEncoderModel` から分けておけば、エンコーダーを差し替えても適合度の計算は変わりません。

`AutoModel.from_pretrained` と `AutoTokenizer.from_pretrained` は、指定したモデル名またはパスから事前学習済みモデルと tokenizer を読み込みます。

コードでは attention mask を用いた mean pooling で embedding を作っています。pooling や正規化の前提は checkpoint ごとに異なるため、検索用の checkpoint を利用するときは、モデルが想定する手法に合わせます。

@[card](https://huggingface.co/docs/transformers/en/models)

`truncation=True` と `max_length` は、入力を指定した長さへ切り詰めます。`max_length` は、利用するモデルが受け付ける最大入力長以下に設定します。文書全体を適合度へ反映する必要がある場合は、文書を分割して複数の適合度を文書単位へ集約する処理を、モデル入力を作る側に追加します。

@[card](https://huggingface.co/docs/transformers/main/en/pad_truncation)

## DataModule でモデル入力を組み立てる

`BiEncoderModel` はクエリと文書のテキストを受け取ります。そこで、データセットの列名や文書ストアの形式は `LightningDataModule` と `DocumentRepository` に閉じ込め、モデルはテキストだけを扱うようにします。

```python
from collections.abc import Mapping

import lightning as L
from torch.utils.data import DataLoader


class DocumentRepository:
    def __init__(self, documents: Mapping[str, str]) -> None:
        self.documents = documents

    def get_texts(self, document_ids: list[str]) -> list[str]:
        return [self.documents[document_id] for document_id in document_ids]


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
            drop_last=True,
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

`DocumentRepository` は、文書 ID から本文を取得する最小のインターフェースです。`get_texts` は受け取った ID と同じ順序で文書のテキストを返します。

`collate_train` は、データセットの record をモデル入力の batch へ変換する collate function です。この順序を保つことで、クエリと正例文書の対応を後続の loss 計算へ渡せます。

このように、DataModule がデータセットの列名と文書ストアを受け持ち、モデルはクエリと文書のテキストだけを扱います。`LightningDataModule` は、データ処理と train／validation／test／predict 用の DataLoader をまとめて管理するクラスです。

@[card](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)

学習用の batch では、クエリと正例文書が同じ順序に並びます。また、後述する in-batch negative を用いるため、各 batch には少なくとも 2 組の正例ペアが必要です。ここでは `drop_last=True` として、末尾が 1 組だけの batch を除いています。この設定では、データ件数が batch size に満たないと学習 batch は生成されません。末尾のデータも毎回使いたい場合は、明示的な負例を batch に加える設計が必要です。

次に、この対応関係を使って loss を計算します。

## 学習モジュールの責務を loss と最適化に絞る

検索モデル固有の学習処理は、適合度と正例の対応から loss を計算する部分です。`training_step` が loss を返し、`configure_optimizers` が optimizer を返すと、Lightning の `Trainer` が学習ループとデバイス配置を実行します。

次の `SearchTrainingModule` では、正例文書がクエリと同じ順序で並ぶことを利用して、対角成分を正例とする loss を計算します。名前に `Trainer` を使わないのは、Lightning では `Trainer` が学習ループを実行する別のオブジェクトを指すためです。

@[card](https://api.lightning.ai/docs/pytorch/stable/common/optimization)

@[card](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)

```python
import torch
import torch.nn.functional as F
from torch import Tensor


class SearchTrainingModule(BiEncoderModel):
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

`compute_loss` は、モデルに batch を渡して適合度を計算し、損失関数の結果を返します。ここでは、正例が対角成分にあることを表す `targets` を作り、cross entropy を計算しています。損失関数の式を変える変更は `compute_loss` に閉じます。一方で、明示的な負例のように新たな入力を加える場合は、`SearchDataModule` も拡張して必要な文書を batch に含めます。

このラベル設定では、同じ batch に含まれる他の文書を in-batch negative として扱います。そのため、あるクエリの正例文書が別のクエリの正例にもなっている場合は、誤った負例（false negative）が含まれます。複数の正例文書や重複文書を含むデータでは、正例の組み合わせを表せる目的関数へ置き換えます。

学習では batch 内の全ペアを比較しました。一方、ランキングではクエリごとに与えられた候補文書だけを比較し、適合度の高い順に並べます。

## ランキングモジュールで予測と結果の書き出しを分ける

ランキングでも `BiEncoderModel` のエンコーダーを使います。ただし、候補文書数はクエリごとに異なるため、学習時のような正方行列ではなく、対応するクエリと文書のペアごとに適合度を計算します。

候補文書を一度フラットなリストにし、各文書がどのクエリに属するかを index で管理すると、この計算を1つの tensor 演算で実装できます。

ここでは、各クエリに少なくとも1件の候補文書があるものとします。

```python
import torch
from torch import Tensor


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

`candidate_counts` の各要素は対応するクエリの候補文書数を表し、`candidate_documents` の要素数はその合計と一致させます。`score_candidates` は、入力した `candidate_documents` と同じ順序で適合度を返します。したがって、候補文書 ID と適合度の対応を保ったまま、結果の整形へ渡せます。

この適合度計算を Lightning の予測処理へつなぐため、`predict_step` から `score_candidates` を呼び出します。`Trainer.predict` は、この `predict_step` を batch ごとに実行します。

```python
from typing import Any

from torch import Tensor


class RankingModule(BiEncoderModel):
    def predict_step(
        self,
        batch: dict[str, Any],
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

適合度をランキング結果へ変換するには、クエリ ID と候補文書 ID も必要です。`RankingDataModule` は、クエリ、候補文書、候補数に加えて、それらの ID を batch に含めます。候補文書はクエリごとのリストをフラット化し、`candidate_counts` で元の対応を保持します。

```python
from typing import Any

import lightning as L
from torch.utils.data import DataLoader


class RankingDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset,
        document_repository: DocumentRepository,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.document_repository = document_repository
        self.batch_size = batch_size

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_predict,
        )

    def collate_predict(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        candidate_id_lists = [record["candidate_document_ids"] for record in records]
        candidate_ids = [
            document_id
            for document_ids in candidate_id_lists
            for document_id in document_ids
        ]
        return {
            "query_ids": [record["query_id"] for record in records],
            "queries": [record["query"] for record in records],
            "candidate_document_ids": candidate_ids,
            "candidate_documents": self.document_repository.get_texts(candidate_ids),
            "candidate_counts": [len(ids) for ids in candidate_id_lists],
        }
```

### 予測を batch 単位で逐次書き出す

`Trainer.predict` の戻り値をすべて保持すると、候補文書数に応じてホストメモリを使います。そこで、`return_predictions=False` を指定し、`PredictionWriter` から予測 batch ごとに結果を書き出します。

```python
from lightning.pytorch.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_path: str) -> None:
        super().__init__(write_interval="batch")
        self.output_path = output_path
        self.temporary_path = f"{output_path}.tmp"
        self.writer = None

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.abort()

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
        self.writer.write(
            query_ids=batch["query_ids"],
            document_ids=batch["candidate_document_ids"],
            candidate_counts=batch["candidate_counts"],
            scores=prediction.detach().cpu(),
        )
```

`PredictionWriter` は、クエリ ID、候補文書 ID、候補数、適合度を `writer` に渡します。`writer.write` の実装では、`candidate_counts` を使って候補文書をクエリごとに復元し、適合度の降順に並べて結果行へ変換します。書き込み先を一時ファイルにしておくと、予測が成功した場合だけ最終パスへ置き換えられます。

```python
query_encoder = TextEncoder("path/to/query-encoder", max_length=256)
document_encoder = TextEncoder("path/to/document-encoder", max_length=256)
model = RankingModule.load_from_checkpoint(
    "search.ckpt",
    query_encoder=query_encoder,
    document_encoder=document_encoder,
)

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

`RankingModule.load_from_checkpoint` に渡すエンコーダーの構成は、学習時と一致させます。この例では `SearchTrainingModule` と `RankingModule` がともに `BiEncoderModel` を継承し、学習パラメータがクエリ用・文書用エンコーダーだけであるため、同じ state dict を復元できます。学習モジュールに追加の学習パラメータを持たせる場合は、学習と予測で同じモジュールを使うか、checkpoint の state dict を対応させます。

`open_result_writer` は一時ファイルへ結果を書き込む writer で、`write`、`close`、`abort` を提供するものとします。`close` は writer を閉じて一時ファイルを最終パスへ置き換え、`abort` は一時ファイルを削除します。

@[card](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html)

この構成では、予測結果を全件保持せずに batch 単位で書き出せます。`BasePredictionWriter` は、batch 単位の予測結果の書き出しを実装する callback です。

なお、この writer は、1つのプロセスが1つの結果ファイルへ書き込む構成を前提にしています。そのため、複数 GPU や複数プロセスで実行する場合は、rank ごとに一時ファイルを分け、最後に merge する設計が必要です。

@[card](https://lightning.ai/docs/pytorch/stable/common/trainer.html)

@[card](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BasePredictionWriter.html)

## 責務の境界

実装上の責務は、次のように分けられます。

| 責務                    | 担当                      | 主な入出力                 |
| ----------------------- | ------------------------- | -------------------------- |
| クエリのベクトル化      | `TextEncoder`（クエリ用） | クエリ → クエリ embedding  |
| 文書のベクトル化        | `TextEncoder`（文書用）   | 文書 → 文書 embedding      |
| 適合度の計算            | `BiEncoderModel`          | embedding → 適合度         |
| loss／目的関数の計算    | `SearchTrainingModule`    | 適合度 → loss              |
| 学習 batch の準備       | `SearchDataModule`        | データセット → モデル入力  |
| ランキング batch の準備 | `RankingDataModule`       | データセット → 予測入力    |
| 予測結果の書き出し      | `PredictionWriter`        | ID + 適合度 → 結果ファイル |

モデル本体は embedding と適合度だけを扱います。そのため、損失関数の式を変える場合は `SearchTrainingModule`、データ形式を変える場合は DataModule、保存形式を変える場合は `PredictionWriter` を変更します。負例などの追加入力は、学習モジュールと DataModule の両方で扱います。学習とランキングで共通の計算を持つ一方で、それぞれの変更が他の処理へ広がりにくくなります。

## まとめ

検索モデルでは、モデルとデータ処理、学習、ランキング結果の保存の境界を定義します。

`BiEncoderModel` にクエリと文書の embedding と適合度を集約し、DataModule からモデル入力だけを渡します。`SearchTrainingModule` では適合度から loss を計算し、`RankingModule` では候補文書の適合度を求めて結果として保存します。

ランキング結果を batch 単位で書き出すことで、予測結果を全件保持せずに処理できます。エンコーダー、目的関数、データ形式、保存形式の変更先が分かれることが、この構成の利点です。
