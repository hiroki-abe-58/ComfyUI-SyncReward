## SyncReward-Bench: 1000 of 1000 clips scored, 0 failed
| metric | this pack | model card (Stage 2) |
| --- | --- | --- |
| spearman | 0.7810 [0.756, 0.805] | 0.7823 |
| top1_n4 | 0.7639 [0.713, 0.821] | 0.7646 |
| pair_acc | 0.8352 [0.822, 0.848] | 0.8359 |
| pearson | 0.7711 |  |
| kendall | 0.6170 |  |
| pair_acc_gap0.75 | 0.9626 |  |
| mae | 0.2860 |  |

chance level for top-1 (N=4): 0.3323; pairs: 421748, pairs with target gap > 0.75: 153238

| generator | clips | Spearman | pair acc | mean target | mean reward |
| --- | --- | --- | --- | --- | --- |
| davinci | 143 | 0.666 | 0.784 | 0.77 | 0.77 |
| kling | 143 | 0.757 | 0.835 | 1.30 | 1.35 |
| ltx2.3 | 143 | 0.789 | 0.845 | 1.25 | 1.23 |
| mova | 143 | 0.684 | 0.797 | 0.85 | 0.91 |
| nava | 143 | 0.734 | 0.819 | 1.07 | 1.06 |
| ovi | 143 | 0.758 | 0.820 | 0.89 | 0.84 |
| seedance | 142 | 0.596 | 0.759 | 1.41 | 1.32 |

Pairwise accuracy by size of the human gap (a tie of the reward counts half):

| human target gap | pairs | reward orders them like the humans |
| --- | --- | --- |
| 0.00 < gap <= 0.34 | 147650 | 0.694 |
| 0.34 < gap <= 0.67 | 120860 | 0.846 |
| 0.67 < gap <= 1.01 | 81398 | 0.941 |
| 1.01 < gap <= 2.01 | 71840 | 0.987 |

Best-of-N selection among random groups of benchmark clips (as in *Select Best*; a group counts as right when the top-scored clip is one of the human-best ones, ties of the reward share the credit; 1000 random groupings):

| N | picks a human-best clip | chance |
| --- | --- | --- |
| 2 | 0.836 | 0.500 |
| 3 | 0.800 | 0.404 |
| 4 | 0.764 | 0.332 |
| 6 | 0.714 | 0.248 |

reward of clips humans rated >= 1.5 (n=260): mean 1.569; rated <= 0.5 (n=192): mean 0.506; P(high-rated clip scores above low-rated clip) = 0.991

## Controlled changes: 70 clips rated >= 1.5 by humans; 0 variants failed
original reward: mean 1.539, SD 0.301, min 0.598, max 1.914

harmless changes (gain -12 dB, high-quality re-encode crf 18 / AAC 192k; 140 pairs): reward drop original - variant has mean +0.0085, SD 0.0521; 95th percentile 0.1098, 99th 0.1829

| change | n | mean reward | mean drop vs original [95% CI] | original higher | P(original > variant) | detected at 5% false alarm | at 1% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shift_-1000ms | 70 | 1.527 | +0.012 [+0.005, +0.020] | 0.53 | 0.510 | 0.03 | 0.00 |
| shift_-500ms | 70 | 1.533 | +0.006 [+0.001, +0.011] | 0.50 | 0.506 | 0.00 | 0.00 |
| shift_-200ms | 70 | 1.538 | +0.001 [-0.004, +0.006] | 0.46 | 0.496 | 0.00 | 0.00 |
| shift_+200ms | 70 | 1.539 | -0.000 [-0.005, +0.005] | 0.36 | 0.501 | 0.00 | 0.00 |
| shift_+500ms | 70 | 1.534 | +0.005 [-0.001, +0.010] | 0.49 | 0.505 | 0.00 | 0.00 |
| shift_+1000ms | 70 | 1.525 | +0.014 [+0.004, +0.025] | 0.57 | 0.511 | 0.01 | 0.01 |
| reversed | 70 | 1.476 | +0.063 [+0.051, +0.076] | 0.83 | 0.587 | 0.19 | 0.00 |
| silence | 70 | 1.283 | +0.256 [+0.183, +0.339] | 0.81 | 0.674 | 0.57 | 0.46 |
| white_noise_same_rms | 70 | 1.283 | +0.255 [+0.209, +0.304] | 0.93 | 0.747 | 0.67 | 0.60 |
| other_clip_same_generator | 70 | 1.414 | +0.124 [+0.069, +0.191] | 0.70 | 0.584 | 0.31 | 0.26 |
| other_clip_other_generator | 70 | 1.405 | +0.134 [+0.044, +0.237] | 0.59 | 0.545 | 0.37 | 0.34 |
| gain_-12dB | 70 | 1.541 | -0.002 [-0.016, +0.012] | 0.43 | 0.491 | 0.04 | 0.01 |
| enc_crf18_aac192 | 70 | 1.520 | +0.019 [+0.010, +0.029] | 0.56 | 0.527 | 0.06 | 0.01 |
| enc_crf30_aac64 | 70 | 1.333 | +0.206 [+0.177, +0.241] | 1.00 | 0.692 | 0.71 | 0.54 |
| enc_half_res | 70 | 1.396 | +0.143 [+0.117, +0.173] | 0.97 | 0.637 | 0.43 | 0.29 |
| enc_fps25 | 70 | 1.469 | +0.070 [+0.052, +0.090] | 0.83 | 0.573 | 0.20 | 0.13 |
| enc_audio16k_mono | 70 | 1.346 | +0.193 [+0.166, +0.221] | 0.99 | 0.699 | 0.69 | 0.56 |
| enc_mpeg4_mp3 | 70 | 1.519 | +0.020 [-0.001, +0.040] | 0.50 | 0.521 | 0.14 | 0.06 |

'original higher' = share of clips whose original reward is above the variant's; P(original > variant) is the same over all pairs of clips (a tie counts half). A selection between the original and a changed copy is wrong when the changed copy scores at least as high.

| generator | clips | mean original | mean drop at 1.0 s (either direction) | mean drop for silence |
| --- | --- | --- | --- | --- |
| davinci | 10 | 1.340 | +0.015 | +0.120 |
| kling | 10 | 1.666 | +0.026 | +0.544 |
| ltx2.3 | 10 | 1.518 | +0.014 | +0.299 |
| mova | 10 | 1.634 | +0.008 | +0.161 |
| nava | 10 | 1.523 | +0.005 | +0.079 |
| ovi | 10 | 1.538 | -0.000 | -0.002 |
| seedance | 10 | 1.552 | +0.022 | +0.589 |
