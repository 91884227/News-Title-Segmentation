# News-Title-Segmentation

## usage
```
python news_segmentation.py FILENAME WORD_TO_WEIGHT LIMIT CUDA_VISIBLE_DEVICES GPU_MEMORY_FRACTION
```
| Parameter | meaning | e.g. |
| -------- | -------- | -------- |
| FILENAME | 處理檔案名稱(.npy) | "katino_data_adjust.npy" |
| WORD_TO_WEIGHT |  詞典(特別關注的詞彙及它們的相對權重)(.txt)| "dictionary.txt"|
| LIMIT | 1: 跑完整個檔案 0: 只處理檔案前1000筆 | 1 |
| CUDA_VISIBLE_DEVICES |  使用的gpu名稱| 0|
| GPU_MEMORY_FRACTION |  gpu使用量的最高使用量 值須介於0到1| 0.7|

執行完後即會輸出 ==FILENAME_ws.json== 和  ==FILENAME_POS.json==

## example
```
python news_segmentation.py "katino_data_adjust.npy" "dictionary.txt" 1 0 0.7
```
執行完後即會輸出 ==katino_data_adjust_ws.json== 和  ==katino_data_adjust.json==

## Remark:
要先將中研院斷詞[中研院斷詞](https://github.com/ckiplab/ckiptagger/wiki/Chinese-README)
的```2. 載入模型```先下載，放在同層目錄 才能執行


