import os
import json

def datafold_read(datalist, basedir, fold, mode):
    with open(datalist) as f:
        json_data = json.load(f)
    
    json_data = json_data[mode]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

                
    tr = []
    val = []
    
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)
    
    return tr, val
