---
title: "polarsで数十万件のデータ処理を高速化"
emoji: "🐻‍❄️"
type: "tech"
topics: ["python", "polars", "machinelearning", "parquet"]
published: true
---

:::details この記事で分かること

- 逐次書き込みと scan_parquet を採用し、データ読込時のメモリ不足（OOM）を回避
- LazyFrame でもメモリを圧迫する collect の判断基準
- collect_batches を採用し、GPU 推論へ渡すデータをバッチ単位に限定
  :::

私は業務委託や研究で、数十万件規模のデータを扱う機械学習モデルを開発しています。モデル開発では、前処理、学習データの作成、評価、推論で同じデータを何度も読みます。データ量が増えるほど、データの持ち方によって処理時間と RAM 使用量が大きく変わります。

当初は、読み込んだレコードを list にためてから DataFrame へ変換していました。データが増えるとメモリ不足（OOM）が発生しました。小分けにすると OOM は避けられましたが、Python で1行ずつ処理し、必要のない列まで何度も読み込むため、処理に時間がかかりました。

## Polars とは

Polars は、構造化データを扱う DataFrame ライブラリです。コアは Rust で実装され、Python から利用できます。

LazyFrame では、処理をすぐに実行せず、実行計画として保持します。実行前に計画全体を最適化するため、不要な列や行の読込を減らせます。ストリーミング実行に対応する処理は、データをバッチ単位で処理できます。

参照: [Polars User Guide](https://docs.pola.rs/user-guide/)、[Polars Lazy API](https://docs.pola.rs/user-guide/concepts/lazy-api/)、[Polars Streaming](https://docs.pola.rs/user-guide/concepts/streaming/)

この特徴を使って処理を見直すと、RAM を使い切りやすい箇所は処理の流れに沿って4つありました。

- XML、JSON、ログから取り出した数十万件のイベントを list にためると、DataFrame を作る前に RAM を消費する
- Parquet へ一定件数ごとに書き込んでも、read_parquet で全列を読むとデータ全体が RAM に載る
- LazyFrame で読み取るデータ量を減らしても、collect の結果が大きければ DataFrame が RAM を圧迫する
- モデル推論では LazyFrame のまま外部モデルへ渡せないため、Python オブジェクトとテンソルへ変換する境界を設計する必要がある

そこで、全件は Parquet に保存し、Python、DataFrame、GPU には必要な範囲だけを置く構成へ変更しました。以降、この4箇所を順に見直します。

コードでは、イベントレコードと参照先エンティティのマスタを処理します。実際の研究用コードから、固有のデータ名や識別子を除いています。

以降のコードは polars 1.43.2 以降が前提です。Parquet をバッチ単位で書き込む箇所では pyarrow 25.0.1 以降も使います。

## 解決の流れ

4つの問題を解決するため、データの保持場所を次のように分けます。

| 保持場所            | 保持する範囲                   | 注意点                            |
| ------------------- | ------------------------------ | --------------------------------- |
| Python オブジェクト | 処理中のバッチ                 | list や dict の管理コストが加わる |
| Polars DataFrame    | 小さい結果または処理中のバッチ | 具体化した結果が RAM に載る       |
| Parquet             | 再利用する全データ             | 読み書きの回数が増えると遅くなる  |
| GPU メモリ          | 推論中のバッチ                 | バッチサイズに強く制約される      |

実装は次の順で進めます。

1. 生データは逐次読み込み、一定件数ごとに Parquet へ保存する
2. Parquet の整形は LazyFrame に積み、読取列・読取行を減らしてから実行する
3. collect は小さい結果、または次の処理が DataFrame を必要とする境界に限定する
4. モデルに渡すデータは collect_batches で分割し、出力もバッチごとに書き出す

## 1. 生データを Parquet へ逐次書き込む

イテレータから得たレコードをそのまま list にためる実装を避けます。

```python
# 避けたい例: records が全件の dict を保持
records = list(iter_events("events.xml"))
pl.DataFrame(records).write_parquet("events.parquet")
```

参照先のリストや大きな文字列を含むレコードでは、Python の dict と list が大きくなります。DataFrame への変換中は、Python 側と列指向データ側の両方を保持します。

入力はイテレータのまま読み、固定件数ごとに Parquet へ書き出します。

```python
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


SCHEMA = {
    "event_id": pl.Int64,
    "event_type": pl.String,
    "partition": pl.String,
    "referenced_entities": pl.List(
        pl.Struct({"entity_id": pl.String, "position": pl.Int64})
    ),
}


def write_parquet_batches(
    records: Iterator[dict[str, object]],
    output_path: Path,
    batch_size: int = 10_000,
) -> None:
    arrow_schema = pl.DataFrame(schema=SCHEMA).to_arrow().schema
    with pq.ParquetWriter(output_path, arrow_schema, compression="zstd") as writer:
        while batch := list(islice(records, batch_size)):
            table = pl.DataFrame(batch, schema=SCHEMA).to_arrow()
            writer.write_table(table)


write_parquet_batches(
    iter_events("events.xml"),
    Path("events_raw.parquet"),
)
```

このコードでも1バッチ分の Python オブジェクトは作ります。ただし保持量は、入力全体ではなく batch_size と1レコードの大きさで概ね決まります。文字列や参照先のリストが大きい場合は、batch_size も小さくします。

これで、生データの読込時に保持する量を1バッチ分に抑えられます。次の課題は、Parquet を全件読み込むと、再びデータ全体が RAM に載ることです。

## 2. LazyFrame で必要なデータだけを読む

Parquet に保存しても、read_parquet でファイル全体を読み込むと全データが DataFrame になります。

```python
# 小規模データや対話的な探索には妥当ですが、全件が具体化されます
events = pl.read_parquet("events_raw.parquet")
validation_events = (
    events.filter(pl.col("partition") == "validation")
    .select("event_id", "referenced_entities")
)
```

即座に DataFrame を得られるため、途中結果を確認する探索作業や、全件が RAM に収まる処理には適しています。

全件を RAM に載せたくない処理では、scan_parquet から LazyFrame を作ります。必要な列と行を絞ってから実行できるためです。参照: [Polars Sources and Sinks](https://docs.pola.rs/user-guide/lazy/sources_sinks/)

イベントが参照するマスタレコードの欠損を検出し、該当イベントを除外します。この処理では、走査、ネスト列の展開、結合、保存を LazyFrame のままつなぎます。

```python
import polars as pl

invalid_event_ids = (
    pl.scan_parquet("events_raw.parquet")
    .select("event_id", "referenced_entities")
    .explode(
        "referenced_entities",
        empty_as_null=False,
        keep_nulls=False,
    )
    .unnest("referenced_entities")
    .drop_nulls("entity_id")
    .join(
        pl.scan_parquet("entity_master.parquet").select("entity_id"),
        on="entity_id",
        how="anti",
    )
    .select("event_id")
    .unique()
)

valid_events = (
    pl.scan_parquet("events_raw.parquet")
    .join(invalid_event_ids, on="event_id", how="anti")
)

valid_events.sink_parquet("events.parquet")
```

explode はネストした参照先を1行ずつに展開するため、行数を増やします。欠損確認だけに使い、event_id を重複除去して元のイベントへ anti join します。展開後のデータは collect せず、そのまま結合に使います。

scan_parquet はこの時点でデータ本体を読まず、実行計画だけを組み立てます。sink_parquet は実行の境界であり、結果を DataFrame に collect せず保存できます。参照: [Polars Sources and Sinks](https://docs.pola.rs/user-guide/lazy/sources_sinks/)

Polars は実行前に計画を最適化します。projection pushdown では必要な列だけを読み、predicate pushdown では可能なフィルタを読込側へ寄せます。参照: [Polars Lazy API](https://docs.pola.rs/user-guide/concepts/lazy-api/)、[Polars Optimizations](https://docs.pola.rs/user-guide/lazy/optimizations/)

次のように計画を確認できます。

```python
validation_events = (
    pl.scan_parquet("events.parquet")
    .filter(pl.col("partition") == "validation")
    .select("event_id", "referenced_entities")
)

print(validation_events.explain())
```

出力には Parquet SCAN、PROJECT、SELECTION などが現れます。表記は Polars のバージョンで変わるため、文字列を固定したテストには向きません。必要な列だけを読んでいるか、SELECTION が Parquet SCAN に含まれているかを確認します。

これで Parquet から読む列と行を減らせます。ただし、最後に collect した結果は DataFrame として RAM に保持されます。

## 3. collect は結果の大きさで判断する

LazyFrame でも、collect の結果は DataFrame です。ストリーミング実行はデータをバッチ単位で処理しますが、返り値の DataFrame は全件保持されます。ストリーミングに対応していない処理は、インメモリエンジンへ切り替わります。参照: [Polars Streaming](https://docs.pola.rs/user-guide/concepts/streaming/)

collect の位置は、次に渡す処理と結果の大きさで決めます。たとえば、学習対象のイベントが参照するエンティティ ID だけを取り出し、十分小さい場合に具体化します。

```python
training_entity_ids = (
    pl.scan_parquet("events.parquet")
    .filter(pl.col("partition") == "training")
    .select("referenced_entities")
    .explode(
        "referenced_entities",
        empty_as_null=False,
        keep_nulls=False,
    )
    .unnest("referenced_entities")
    .drop_nulls("entity_id")
    .select("entity_id")
    .unique()
    .collect(engine="streaming")
    .get_column("entity_id")
)
```

training_entity_ids も RAM を使います。エンティティ ID のユニーク数が大きすぎるなら、この段階で全件を具体化しません。

ID の一覧と抽出後のレコードが RAM に収まる場合は、マスタから必要な2列だけを抜き出します。

```python
entities_for_training = (
    pl.scan_parquet("entity_master.parquet")
    .filter(pl.col("entity_id").is_in(training_entity_ids))
    .select("entity_id", "feature_values")
    .collect(engine="streaming")
)
```

この collect が外部ライブラリとの境界です。Python の list や NumPy 配列へ変換する場合は、変換後のデータも含めて RAM に収まるか確認します。

ここまでは、外部ライブラリへ渡すデータが RAM に収まる場合の処理です。評価データ全体を GPU で推論する場合は、入力から出力までをバッチ単位に分けます。

## 4. Python と GPU の境界をバッチに閉じる

モデル推論では、PyTorch などへ Python オブジェクトやテンソルを渡します。LazyFrame のまま GPU へは渡せません。

全評価データを collect し、参照先マスタをすべて dict にすると、Python オブジェクト、DataFrame、GPU テンソルが重なります。collect_batches で評価イベントを取り出し、そのバッチが参照するエンティティだけを読みます。この構造は、関連データを集めて外部モデルやライブラリへ渡す処理に使えます。

```python
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


def predict_all(
    events: pl.LazyFrame,
    entities_path: Path,
    output_path: Path,
    batch_size: int,
) -> None:
    with pq.ParquetWriter(output_path, schema=RESULT_SCHEMA) as writer:
        for batch in events.collect_batches(
            chunk_size=batch_size,
            engine="streaming",
        ):
            records = list(batch.iter_rows(named=True))

            referenced_entity_ids = (
                batch.select("referenced_entities")
                .explode(
                    "referenced_entities",
                    empty_as_null=False,
                    keep_nulls=False,
                )
                .unnest("referenced_entities")
                .get_column("entity_id")
                .drop_nulls()
                .unique()
            )

            entities = (
                pl.scan_parquet(entities_path)
                .filter(pl.col("entity_id").is_in(referenced_entity_ids))
                .select("entity_id", "feature_values")
                .collect(engine="streaming")
            )
            features_by_id = dict(entities.iter_rows())

            predictions = predict(records, features_by_id)  # GPU 推論
            writer.write_table(to_result_table(batch, predictions))


evaluation_events = (
    pl.scan_parquet("events.parquet")
    .filter(pl.col("partition") == "validation")
    .select("event_id", "referenced_entities")
)

predict_all(
    events=evaluation_events,
    entities_path=Path("entity_master.parquet"),
    output_path=Path("predictions.parquet"),
    batch_size=1_024,
)
```

RESULT_SCHEMA は出力列、predict は推論、to_result_table は Arrow 形式への変換を担います。保持する単位は次のとおりです。

1. batch は評価イベントの一部だけ
2. records と features_by_id は、そのバッチが参照する分だけ
3. GPU に載せる入力と中間テンソルは1バッチ分だけ
4. 予測値は1バッチ分ずつ書き出す

chunk_size は、1回に返すバッチの行数です。処理全体のメモリ上限ではありません。RAM と GPU メモリの両方を見て調整します。参照先の数や特徴量の大きさにばらつきがあれば、同じ行数でも必要なメモリは変わります。

この構成では、バッチごとに参照先マスタを走査します。マスタを RAM に保持できる場合は最初に読み込み、保持できない場合はファイルの分割やキャッシュも検討します。

collect_batches は仕様が安定しておらず、sink_parquet など Polars 内で完結する出力処理より低速です。Python や GPU へ渡す処理に限定し、Polars だけで完結する処理には sink_parquet を使います。Polars のバージョンを固定し、最大級の実データバッチを含む結合テストも用意します。参照: [polars.LazyFrame.collect_batches](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect_batches.html)

これで、外部モデルへ渡す Python オブジェクト、GPU テンソル、出力をバッチ単位に抑えられます。最後に、この構成が向く条件を整理します。

## Polars を採用する判断基準

Polars がすべてのデータ処理に適しているとは限りません。次のように使い分けます。

| 状況                           | 採用しやすい理由                           | 注意点                                   |
| ------------------------------ | ------------------------------------------ | ---------------------------------------- |
| Parquet を何度も絞り込む前処理 | LazyFrame が読取列・読取行を減らせる       | collect が早すぎると利点を失う           |
| ネストしたログの結合・集約     | explode、unnest、join を式として構成できる | 展開後の行数を意識する                   |
| CPU / GPU 推論の入力作成       | LazyFrame とバッチ処理の境界を明示できる   | Python・GPU への変換はバッチ内に限定する |
| 小規模な探索                   | read_parquet で簡潔に書ける                | 無理に LazyFrame だけへ統一しない        |
| pandas 前提の評価ライブラリ    | 直前まで Polars で絞ってから変換できる     | 全件 to_pandas のサイズを確認する        |

Polars を使っても、処理の大半を自前実装の関数が占めれば高速化は期待できません。map_elements は Polars の式より遅くなりやすいため、まず select や with_columns の中で、文字列式、リスト式、構造体式を組み合わせます。自前実装が必要な場合だけ対象を絞って使います。参照: [polars.Expr.map_elements](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.map_elements.html)

## まとめ

数十万件の処理では、Polars の採用自体より、全件をどこに保持するかが重要です。

- 生データは逐次読み込み、Parquet を再利用可能な境界にする
- 前処理は scan_parquet と LazyFrame に積み、explain で読み取る列と行を確認する
- collect は小さい結果、または外部処理へ渡す直前に限定する
- 推論時の Python オブジェクト、GPU テンソル、出力は collect_batches を使ってバッチに閉じる

Parquet を中心に前処理・評価・推論をつなぐ機械学習パイプラインでは、必要なデータだけを読み、外部処理との境界でバッチ化できることが Polars を採用する理由です。
