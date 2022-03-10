import numpy as np
import pandas as pd
import requests as req
import json
# %config Completer.use_jedi = False

url = "https://recruitment.aimtechnologies.co/ai-tasks"
ids_df = pd.read_csv("../input/dialect-ids/dialect_dataset.csv")

txt_df = pd.DataFrame()
j = 0
    
for i in range(1000, len(ids) + 1000, 1000):
    
    if i > len(ids) + 1 : 
        i = i - i % (len(ids) + 1)

    r = req.post(url, data=json.dumps(ids[j : i]))
    
    ids_ = [int(inp) for inp, _ in r.json().items()]
    txt = [inp for _, inp in r.json().items()]
    dic = {"ID" : ids_, "Text": txt}
    
    tmp = pd.DataFrame(dic)
    txt_df = txt_df.append(tmp)
    j = i

full_df = txt_df.merge(ids_df, left_on="ID", right_on="id").drop(["ID", "id"], axis=1)
full_df.head()

full_df.to_csv("full_df.csv", encoding="utf-8-sig", index=False)
