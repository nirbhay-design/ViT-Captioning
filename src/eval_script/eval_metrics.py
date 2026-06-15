# class - bleu score, rouge score, cider score
import torch
import torchmetrics
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BLEUScore
# from aac_metrics import evaluate

def get_metrics(candidates, references):
    bs1 = BLEUScore(n_gram = 1)
    bs2 = BLEUScore(n_gram = 2)
    bs3 = BLEUScore(n_gram = 3)
    bs4 = BLEUScore(n_gram = 4)
    rouge = ROUGEScore()

    output_r = rouge(candidates, references)
    output1 = bs1(candidates, references)
    output2 = bs2(candidates, references)
    output3 = bs3(candidates, references)
    output4 = bs4(candidates, references)
    # scores, _ = evaluate(candidates, references, metrics=["cider_d", "meteor", "spice"])

    output_dict = {
        "rouge1": round(output_r["rouge1_fmeasure"].item(), 4),
        "rouge2": round(output_r["rouge2_fmeasure"].item(), 4),
        "rougeL": round(output_r["rougeL_fmeasure"].item(), 4),
        "blue1": round(output1.item(), 4),
        "blue2": round(output2.item(), 4),
        "blue3": round(output3.item(), 4),
        "blue4": round(output4.item(), 4),
    }

    return output_dict 

if __name__ == "__main__":
    candidates = ["a dog running outside", "a person sitting on a chair"]
    references = [["a dog is running outdoors", "a brown dog runs on the grass"], ["one person sitting on chair"]]

    print(get_metrics(candidates, references))