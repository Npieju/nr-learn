# Next Issue: Class Rest Surface Interaction Strengthening

## Summary

`#40` で runtime 後の experiment queue を `tighter policy search` に戻して artifact を読み直した結果、same-family widening をさらに続ける理由は弱いと判断した。次の primary experiment line は、feature ranking の Tier A / Rank 1 である `class / rest / surface change` family に移す。

## Objective

current high-coverage baseline に残っている `class / rest / surface` family を interaction 単位で強化し、September difficult regime と broad latest benchmark の両方で説明しやすい alpha を狙う。

## In Scope

- `horse_days_since_last_race`
- `horse_weight_change`
- `horse_distance_change`
- `horse_surface_switch`
- `race_class_score`
- `horse_class_change`
- 上記 family の interaction / bucket / regime-aware feature

## Non-Goals

- pedigree-heavy expansion
- low-coverage pace-fit expansion
- policy family widening の再開
- NAR work

## Success Criteria

- feature family が Tier A ranking に沿っている
- high-coverage baseline を壊さず interaction gain を狙う hypothesis になっている
- next formal candidate までつながる artifact plan がある

## Starting Context

current read:

- `tighter policy search` reentry では A anchor 維持の結論が再確認された
- Candidate C は near-par challenger reference に留まり、Candidate B は no-op / failed side read
- policy widening より、baseline family の alpha source を掘るほうが新規性が高い
- feature ranking では `class / rest / surface change` が最上位

## Suggested Validation

- baseline feature inventory refresh
- reduced smoke / evaluation on interaction candidate
- if promising, formal revision gate
