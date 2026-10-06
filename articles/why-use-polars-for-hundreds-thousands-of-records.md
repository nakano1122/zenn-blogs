---
title: "polarsで数十万件のデータ処理を高速化"
emoji: "🐻‍❄️"
type: "tech"
topics: ["python", "polars", "machinelearning", "parquet"]
published: true
---

私は業務委託や研究で、数十万件規模のデータを使い、機械学習モデルを開発しています。モデル開発では、前処理、学習データの作成、評価、推論で同じデータを何度も読み込みます。工程ごとに全件を読み直すと、RAM を圧迫するだけでなく、不要な列や行を読む時間も積み重なります。

当初は、読み込んだレコードを list にため、まとめて DataFrame へ変換していました。データ量が増えると、変換前の Python オブジェクトと変換後の列データが同時に RAM を使うため、メモリ不足（OOM）が発生しました。レコードを小分けにすると OOM は避けられましたが、Python で1行ずつ処理し、使わない列まで何度も読むため、今度は処理時間が問題になりました。

:::details この記事で分かること

- 逐次書き込みと scan_parquet を採用し、データ読込時のメモリ不足（OOM）を回避
- 結果を全件保持する collect の判断基準
- collect_batches を採用し、GPU 推論へ渡すデータをバッチ単位に限定
  :::

## Polars とは

この OOM と処理時間の問題を解決するために、Polars を採用しました。Polars は構造化データを扱う DataFrame ライブラリで、Rust 製のコアを Python から利用できます。

LazyFrame は、処理をすぐに実行せず、実行計画として保持する仕組みです。実行前に計画全体を最適化するため、不要な列や行の読込を減らせます。また、ストリーミング実行に対応する処理では、データのバッチ処理が可能です。

参照: [Polars User Guide](https://docs.pola.rs/user-guide/)、[Polars Lazy API](https://docs.pola.rs/user-guide/concepts/lazy-api/)、[Polars Streaming](https://docs.pola.rs/user-guide/concepts/streaming/)

Parquet と Polars の遅延実行を組み合わせると、処理が進むにつれて現れるメモリ上の課題を段階的に狭められます。最初の対策は、生データを list にためない Parquet への逐次書き込みです。ただし、保存先を変えただけでは、読み込み時に全件が RAM に載るという課題が残ります。LazyFrame で必要な列と行を絞っても、collect すれば結果全体が DataFrame になるため、最後は外部ライブラリや GPU へ渡す単位の制限が必要です。

この流れに合わせて、全件は Parquet に保存し、Python オブジェクト、Polars DataFrame、GPU メモリには処理中のデータだけを置く構成へ変更しました。

以降は、この構成をイベントレコードと参照先エンティティのマスタで示します。基にしたのは実際の研究用コードですが、固有のデータ名や識別子は取り除きました。

以降のコードは polars 1.43.2 以降が前提で、Parquet をバッチ単位で書き込む箇所では pyarrow 25.0.1 以降も使います。

## データの保持範囲を決める

ピーク時の RAM 使用量を左右するのは、同時に保持するデータの量です。そこで、再利用する全件データは Parquet に置き、RAM と GPU メモリには次の処理に必要な範囲だけを読み込みます。

| 保持場所            | 保持する範囲                   | 注意点                            |
| ------------------- | ------------------------------ | --------------------------------- |
| Python オブジェクト | 処理中のバッチ                 | list や dict の管理コストが加わる |
| Polars DataFrame    | 小さい結果または処理中のバッチ | 具体化した結果が RAM に載る       |
| Parquet             | 再利用する全データ             | 読み書きの回数が増えると遅くなる  |
| GPU メモリ          | 推論中のバッチ                 | バッチサイズに強く制約される      |

最初の対象は、DataFrame を作る前の Python オブジェクトです。この段階で全件を保持すると、後続の処理を最適化する前に OOM が発生するためです。

## 1. 生データを Parquet へ逐次書き込む

イテレータから得たレコードを list にためると、DataFrame を作る前に全件分の Python オブジェクトが RAM に載ります。

```python
# 避けたい例: records が全件の dict を保持
records = list(iter_events("events.xml"))
pl.DataFrame(records).write_parquet("events.parquet")
```

参照先のリストや大きな文字列を含むほど、各 dict が使う RAM は増えます。さらに、DataFrame への変換中は、変換前の Python オブジェクトと変換後の列データを二重に保持する状態です。

この重複を避けるため、入力はイテレータのまま読み、固定件数ごとに Parquet へ書き出します。

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

このコードでも1バッチ分の Python オブジェクトは残ります。ただし、保持量の目安は入力全体ではなく、batch_size と1レコードの大きさです。1レコードが大きいほど同じ件数でも RAM を使うため、文字列や参照先のリストが大きい場合は batch_size の調整が必要です。

これで、生データの読込時に保持する量を1バッチ分に抑えられます。ただし、解消できるのは保存時の OOM です。後続処理で Parquet を全件読み込めば、再びデータ全体が RAM に載るという課題が残ります。

## 2. LazyFrame で必要なデータだけを読む

前節で全件を Parquet へ退避しましたが、read_parquet でファイル全体を読み込むと全データが DataFrame になります。

```python
# 小規模データや対話的な探索には妥当ですが、全件が具体化されます
events = pl.read_parquet("events_raw.parquet")
validation_events = (
    events.filter(pl.col("partition") == "validation")
    .select("event_id", "referenced_entities")
)
```

即座に DataFrame を得られるため、途中結果を確認する探索作業や、全件が RAM に収まる処理には適しています。

全件を RAM に載せたくない処理では、scan_parquet から LazyFrame を作ります。処理をすぐに実行しないため、必要な列と行を絞る操作まで含めて最適化できるからです。参照: [Polars Sources and Sinks](https://docs.pola.rs/user-guide/lazy/sources_sinks/)

ここで除外するのは、参照先のマスタレコードが欠けているイベントです。欠損の検出ではネスト列の展開によって行数が増えるため、展開後のデータを DataFrame として保持すると RAM を圧迫します。そこで、走査から保存までを LazyFrame のままつなぐ構成にします。

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

explode は、ネストした参照先を1行ずつに分ける操作です。展開後は行数が増えるため、用途を欠損確認に限定します。欠損がある event_id を重複除去して元のイベントへ anti join すれば、展開結果の collect は不要です。

scan_parquet はデータ全件を DataFrame にせず、実行計画を組み立てる入口です。sink_parquet はその計画を実行し、結果を DataFrame に collect せず保存します。参照: [Polars Sources and Sinks](https://docs.pola.rs/user-guide/lazy/sources_sinks/)

LazyFrame のままつないだ処理は、実行前に計画全体を最適化する対象になります。projection pushdown は必要な列だけを読み、predicate pushdown は可能なフィルタを読込側へ寄せる最適化です。参照: [Polars Lazy API](https://docs.pola.rs/user-guide/concepts/lazy-api/)、[Polars Optimizations](https://docs.pola.rs/user-guide/lazy/optimizations/)

最適化が意図どおりに働くかは、explain で実行前に確認できます。

```python
validation_events = (
    pl.scan_parquet("events.parquet")
    .filter(pl.col("partition") == "validation")
    .select("event_id", "referenced_entities")
)

print(validation_events.explain())
```

出力には Parquet SCAN、PROJECT、SELECTION などが現れます。Polars のバージョンによって表記が変わるため、文字列を固定したテストには不向きです。ここで確認するのは、必要な列だけを読み、SELECTION が Parquet SCAN に含まれているかという点です。

これで Parquet から読む列と行を減らせます。ただし、LazyFrame が抑えるのは実行途中の無駄な読み込みです。最後に collect した結果は DataFrame になるため、その大きさによっては RAM を圧迫します。

## 3. collect は結果の大きさで判断する

前節の処理を LazyFrame にしても、collect の結果は DataFrame です。ストリーミング実行でバッチ単位になるのは途中の処理に限られ、返り値の DataFrame には全件が保持されます。また、ストリーミングに対応していない処理がインメモリエンジンへ切り替わる点にも注意が必要です。参照: [Polars Streaming](https://docs.pola.rs/user-guide/concepts/streaming/)

そのため、collect は返される結果が RAM に収まり、次の処理が DataFrame や Python オブジェクトを必要とする位置で実行します。たとえば、学習対象のイベントが参照するエンティティ ID は、重複除去後の件数が十分小さい場合に限って具体化します。

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

イベント数が多くても、参照するエンティティ ID の種類が少なければ、重複除去後の結果は小さくなります。一方、ユニーク数が大きい場合は training_entity_ids 自体が RAM を圧迫します。その場合の選択肢は、ID を collect せずに LazyFrame のまま結合するか、後述するバッチ単位での抽出です。

ID の一覧と抽出後のレコードが RAM に収まる場合は、マスタから必要な2列だけを抜き出します。

```python
entities_for_training = (
    pl.scan_parquet("entity_master.parquet")
    .filter(pl.col("entity_id").is_in(training_entity_ids))
    .select("entity_id", "feature_values")
    .collect(engine="streaming")
)
```

この collect が外部ライブラリとの境界です。ここから Python の list や NumPy 配列へ変換すると、DataFrame と変換後のデータを同時に保持する場合があります。必要な空き容量は抽出結果だけでなく、変換後のデータまで含めて判断します。

ここまでは、外部ライブラリへ渡すデータが RAM に収まる場合の処理です。評価データ全体が収まらない場合は collect の位置を遅らせるだけでは解決できないため、入力から出力までをバッチ単位に分けます。

## 4. Python と GPU の境界をバッチに閉じる

モデル推論では、PyTorch などへ Python オブジェクトやテンソルを渡すため、LazyFrame のまま GPU へは渡せません。

全評価データを collect し、参照先マスタをすべて dict にすると、Python オブジェクト、DataFrame、GPU テンソルを同時に保持する状態です。この重なりを避けるため、collect_batches で評価イベントを取り出し、そのバッチが参照するエンティティだけを読みます。バッチの推論結果をすぐに書き出せば、次のバッチへ進む前に不要なオブジェクトを解放できます。

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

RESULT_SCHEMA は出力列、predict は推論、to_result_table は Arrow 形式への変換を担います。入力の取得から出力までをループ内に置くことで、保持する単位は次のようになります。

1. batch は評価イベントの一部だけ
2. records と features_by_id は、そのバッチが参照する分だけ
3. GPU に載せる入力と中間テンソルは1バッチ分だけ
4. 予測値は1バッチ分ずつ書き出す

chunk_size は、1回に返すバッチの行数です。処理全体のメモリ上限を表す値ではありません。調整の基準は RAM と GPU メモリの両方であり、参照先の数や特徴量の大きさにばらつきがあれば、同じ行数でも必要なメモリは変わります。

この構成はピーク時の保持量を抑える一方で、バッチごとに参照先マスタを走査します。そのため、マスタを RAM に保持できる場合は最初に読み込み、保持できない場合はファイルの分割やキャッシュが選択肢です。

collect_batches は仕様が安定しておらず、sink_parquet など Polars 内で完結する出力処理より低速です。そのため、Python や GPU へ渡す処理に限定し、Polars だけで完結する処理には sink_parquet を使います。バージョンの固定に加えて、最大級の実データバッチを含む結合テストも必要です。参照: [polars.LazyFrame.collect_batches](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect_batches.html)

この結果、外部モデルへ渡す Python オブジェクト、GPU テンソル、出力はバッチ単位に収まります。ただし、バッチ化には走査や変換のコストもあるため、採用の基準はデータの規模と後続処理です。

## Polars を採用する判断基準

前節までの構成が効果的なのは、全件を RAM に保持できず、Parquet の絞り込みや結合を繰り返す場合です。一方、データが小さい場合や処理の大半が Python 側にある場合は、遅延実行やバッチ化で増える実装上の複雑さに対して、得られる効果は小さくなります。この差が、データ量と処理内容に応じて使い分ける基準です。

| 状況                           | 採用しやすい理由                           | 注意点                                   |
| ------------------------------ | ------------------------------------------ | ---------------------------------------- |
| Parquet を何度も絞り込む前処理 | LazyFrame が読取列・読取行を減らせる       | collect が早すぎると利点を失う           |
| ネストしたログの結合・集約     | explode、unnest、join を式として構成できる | 展開後の行数を意識する                   |
| CPU / GPU 推論の入力作成       | LazyFrame とバッチ処理の境界を明示できる   | Python・GPU への変換はバッチ内に限定する |
| 小規模な探索                   | read_parquet で簡潔に書ける                | 無理に LazyFrame だけへ統一しない        |
| pandas 前提の評価ライブラリ    | 直前まで Polars で絞ってから変換できる     | 全件 to_pandas のサイズを確認する        |

処理の大半を自前実装の関数にすると、Polars が最適化できる範囲から外れるため、高速化は期待できません。特に map_elements は Polars の式より遅くなりやすいため、優先するのは select や with_columns の中で文字列式、リスト式、構造体式を組み合わせる実装です。自前実装の関数は、必要な行まで絞った後に使います。参照: [polars.Expr.map_elements](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.map_elements.html)

## まとめ

数十万件の処理では、Polars を採用するだけで OOM を防げるわけではありません。生データを全件保持すると DataFrame を作る前に RAM を使い切るため、最初の対策は Parquet への逐次書き込みです。しかし、Parquet を全件読み込めば同じ問題が再発します。そこで、scan_parquet と LazyFrame によって、必要な列と行だけに読込範囲を絞ります。それでも collect の結果は RAM に載るため、実行するのは結果が十分小さい場合か、外部処理へ渡す直前だけです。外部処理へ渡すデータも大きい場合は、collect_batches で入力、Python オブジェクト、GPU テンソル、出力をバッチ内に閉じます。

scan_parquet で不要な列と行の読み込みを減らし、処理を Polars の式でつなぐと、Python で1行ずつ処理する範囲も減らせます。そのうえで、前の対策で残ったメモリ上の境界を次の対策で狭められることが、Parquet を中心とした機械学習パイプラインで Polars を採用する理由です。
