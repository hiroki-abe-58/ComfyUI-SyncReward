## SyncReward-Bench: 1000 of 1000 clips scored, 0 failed
| metric | this pack | model card (Stage 2) |
| --- | --- | --- |
| spearman | 0.7801 [0.755, 0.804] | 0.7823 |
| top1_n4 | 0.7639 [0.713, 0.820] | 0.7646 |
| pair_acc | 0.8347 [0.821, 0.848] | 0.8359 |
| pearson | 0.7707 |  |
| kendall | 0.6163 |  |
| pair_acc_gap0.75 | 0.9622 |  |
| mae | 0.2859 |  |

chance level for top-1 (N=4): 0.3323; pairs: 421748, pairs with target gap > 0.75: 153238

| generator | clips | Spearman | pair acc | mean target | mean reward |
| --- | --- | --- | --- | --- | --- |
| davinci | 143 | 0.660 | 0.780 | 0.77 | 0.78 |
| kling | 143 | 0.757 | 0.836 | 1.30 | 1.35 |
| ltx2.3 | 143 | 0.789 | 0.845 | 1.25 | 1.23 |
| mova | 143 | 0.684 | 0.797 | 0.85 | 0.91 |
| nava | 143 | 0.735 | 0.819 | 1.07 | 1.07 |
| ovi | 143 | 0.758 | 0.820 | 0.89 | 0.84 |
| seedance | 142 | 0.592 | 0.757 | 1.41 | 1.32 |

Pairwise accuracy by size of the human gap (a tie of the reward counts half):

| human target gap | pairs | reward orders them like the humans |
| --- | --- | --- |
| 0.00 < gap <= 0.34 | 147650 | 0.694 |
| 0.34 < gap <= 0.67 | 120860 | 0.845 |
| 0.67 < gap <= 1.01 | 81398 | 0.941 |
| 1.01 < gap <= 2.01 | 71840 | 0.987 |

Best-of-N selection among random groups of benchmark clips (as in *Select Best*; a group counts as right when the top-scored clip is one of the human-best ones, ties of the reward share the credit; 1000 random groupings):

| N | picks a human-best clip | chance |
| --- | --- | --- |
| 2 | 0.835 | 0.500 |
| 3 | 0.799 | 0.404 |
| 4 | 0.764 | 0.332 |
| 6 | 0.714 | 0.248 |

reward of clips humans rated >= 1.5 (n=260): mean 1.569; rated <= 0.5 (n=192): mean 0.506; P(high-rated clip scores above low-rated clip) = 0.991

## Controlled changes: 70 clips rated >= 1.5 by humans; 0 variants failed
original reward: mean 1.540, SD 0.299, min 0.645, max 1.914

harmless changes (gain -12 dB, high-quality re-encode crf 18 / AAC 192k; 140 pairs): reward drop original - variant has mean +0.0093, SD 0.0537; 95th percentile 0.1176, 99th 0.1746

| change | n | mean reward | mean drop vs original [95% CI] | original higher | P(original > variant) | detected at 5% false alarm | at 1% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shift_-1000ms | 70 | 1.528 | +0.013 [+0.006, +0.020] | 0.54 | 0.511 | 0.00 | 0.00 |
| shift_-500ms | 70 | 1.533 | +0.007 [+0.002, +0.012] | 0.51 | 0.507 | 0.00 | 0.00 |
| shift_-200ms | 70 | 1.539 | +0.001 [-0.004, +0.006] | 0.36 | 0.493 | 0.00 | 0.00 |
| shift_+200ms | 70 | 1.538 | +0.002 [-0.002, +0.007] | 0.47 | 0.504 | 0.00 | 0.00 |
| shift_+500ms | 70 | 1.535 | +0.005 [-0.000, +0.011] | 0.51 | 0.509 | 0.00 | 0.00 |
| shift_+1000ms | 70 | 1.526 | +0.014 [+0.004, +0.026] | 0.60 | 0.516 | 0.01 | 0.01 |
| reversed | 70 | 1.476 | +0.065 [+0.052, +0.078] | 0.87 | 0.589 | 0.14 | 0.01 |
| silence | 70 | 1.284 | +0.256 [+0.183, +0.341] | 0.81 | 0.674 | 0.59 | 0.47 |
| white_noise_same_rms | 70 | 1.284 | +0.256 [+0.209, +0.304] | 0.94 | 0.751 | 0.69 | 0.60 |
| other_clip_same_generator | 70 | 1.419 | +0.121 [+0.067, +0.186] | 0.70 | 0.585 | 0.29 | 0.26 |
| other_clip_other_generator | 70 | 1.407 | +0.133 [+0.045, +0.235] | 0.59 | 0.545 | 0.36 | 0.36 |
| gain_-12dB | 70 | 1.542 | -0.002 [-0.015, +0.013] | 0.40 | 0.489 | 0.04 | 0.01 |
| enc_crf18_aac192 | 70 | 1.520 | +0.020 [+0.011, +0.031] | 0.57 | 0.524 | 0.06 | 0.01 |
| enc_crf30_aac64 | 70 | 1.334 | +0.206 [+0.176, +0.241] | 1.00 | 0.693 | 0.67 | 0.53 |
| enc_half_res | 70 | 1.398 | +0.143 [+0.116, +0.173] | 0.97 | 0.633 | 0.41 | 0.27 |
| enc_fps25 | 70 | 1.470 | +0.071 [+0.052, +0.091] | 0.87 | 0.573 | 0.21 | 0.14 |
| enc_audio16k_mono | 70 | 1.346 | +0.195 [+0.169, +0.221] | 0.99 | 0.699 | 0.69 | 0.56 |
| enc_mpeg4_mp3 | 70 | 1.516 | +0.024 [+0.004, +0.045] | 0.51 | 0.526 | 0.13 | 0.10 |

'original higher' = share of clips whose original reward is above the variant's; P(original > variant) is the same over all pairs of clips (a tie counts half). A selection between the original and a changed copy is wrong when the changed copy scores at least as high.

| generator | clips | mean original | mean drop at 1.0 s (either direction) | mean drop for silence |
| --- | --- | --- | --- | --- |
| davinci | 10 | 1.348 | +0.013 | +0.121 |
| kling | 10 | 1.659 | +0.026 | +0.535 |
| ltx2.3 | 10 | 1.518 | +0.014 | +0.299 |
| mova | 10 | 1.634 | +0.008 | +0.161 |
| nava | 10 | 1.531 | +0.005 | +0.087 |
| ovi | 10 | 1.538 | -0.000 | -0.002 |
| seedance | 10 | 1.555 | +0.027 | +0.592 |
