import pandas as pd
import sys
if len (sys.argv) < 3 :
    print("Usage: python pathtodata pathtooutput")
    sys.exit(1)

import pandas as pd
test=pd.read_csv(sys.argv[1])

from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("../data/data/bidaf-model-2017.09.15-charpad.tar.gz")
i=0
single={}
for index,row in test.iterrows():
  ans=predictor.predict(passage=row['contexts'],question=row['questions'])
  single[row['qids']]=ans['best_span_str']
  i+=1
  print(i,"Done!", end="\r")

import json
with open(sys.argv[2], "w") as write_file:
    json.dump(single, write_file)
print("saved results to",sys.argv[2])