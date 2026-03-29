# Seasonal De-Risk Decision Standard

## 1. Purpose

この文書は、`seasonal de-risk` family を broad replacement と誤読せず、`controlled override` として一貫して扱うための実務標準である。

対象は主に次の 3 本である。

- `current_long_horizon_serving_2025_latest`
- `current_tighter_policy_search_candidate_2025_latest`
- `current_recommended_serving_2025_recent_2018`

このうち最初の 1 本は operational alias、後ろ 2 本は analysis-first candidate として扱う。

## 2. Default Reading

2026-03-29 時点の default reading は次で固定する。

1. baseline は `current_recommended_serving_2025_latest`
2. September difficult window の最初の de-risk alias は `current_long_horizon_serving_2025_latest`
3. `current_tighter_policy_search_candidate_2025_latest` は second defensive option
4. recent-2018 true retrain は third analysis-first fallback

重要なのは、`formal pass / promote` と `operational default` を分けることである。

## 3. Decision Rule

seasonal family は次の 3 択で判断する。

- `operational seasonal alias`
  - September difficult window で baseline より損失圧縮が明確
  - non-September control window では broad replacement にならない
  - policy 差分が局所的で説明しやすい
- `analysis-first defensive candidate`
  - difficult window では強いが、control window で baseline 優位を崩す
  - あるいは retrain / broader policy rewrite を伴い、運用コストが高い
- `reject as seasonal default`
  - difficult window でも優位が弱い
  - または non-target regime を壊す

## 4. Required Read Order

seasonal de-risk 判断は、毎回次の順で読む。

1. September difficult window の dashboard summary
2. December tail control window の dashboard summary
3. 必要なら compare rerun artifact
4. 必要なら formal support artifact

この順序を飛ばして、いきなり promotion gate や single window の勝ち負けだけで運用判断しない。

## 5. Current Operational Read

既存 artifact の current read は次のとおりである。

### 5.1 Long-Horizon

- September 8 日:
  - baseline `-27.3`
  - long-horizon `-4.3`
- December tail 8 日:
  - baseline `+14.9`
  - long-horizon `+14.9`
- differing policy dates:
  - September `8/8`
  - December `0`

読み:

`current_long_horizon_serving_2025_latest` は broad rewrite ではなく、September-only に近い controlled override である。したがって current seasonal de-risk alias はこれを first choice に置く。

### 5.2 Tighter Policy Candidate

- September 8 日:
  - baseline `32 bets / -27.3 / 0.2959`
  - tighter policy `9 bets / -4.3 / 0.8395`
- December tail 8 日:
  - baseline `45 bets / +21.8 / 1.6712`
  - tighter policy `9 bets / +21.4 / 1.6032`

読み:

September difficult window では defensive option として意味があるが、December control では baseline top line を明確には超えない。したがって first seasonal alias ではなく、second defensive option として扱う。

### 5.3 Recent-2018 True Retrain

- September 8 日:
  - baseline `32 bets / -27.3 / 0.2959`
  - recent-2018 `4 bets / -4.0 / 0.8557`
- December tail 8 日:
  - baseline `45 bets / +21.8 / 1.6712`
  - recent-2018 `1 bet / -1.0 / 0.9722`

読み:

September de-risk は強いが、学習窓再構成を伴い、December control では baseline に劣後する。したがって operational alias ではなく third analysis-first fallback とする。

## 6. Standard Conclusion Template

issue や docs には、次の文面に寄せて残す。

```md
Conclusion
- baseline default remains `current_recommended_serving_2025_latest`
- first seasonal de-risk alias remains `current_long_horizon_serving_2025_latest`
- `current_tighter_policy_search_candidate_2025_latest` stays analysis-first second defensive option
- recent-2018 true retrain stays analysis-first third fallback
```

secondary fallback ordering を固定するときは、`docs/seasonal_secondary_fallback_standard.md` を併用する。

## 7. Stop Conditions

次のいずれかが出たら、この標準を見直す。

- long-horizon が control window でも baseline と継続的に乖離する
- tighter policy candidate が September と control の両方で long-horizon を明確に上回る
- recent-heavy retrain が retrain cost を上回る operational優位を示す
- seasonal override が September 以外にも恒常的に必要だと判明する
