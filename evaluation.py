from pathlib import Path
import numpy as np
import pandas as pd
import json
import logging
import re
from sys import stderr, exit as sysexit

logger=logging.getLogger(__name__)

DEFAULT_EVALUATION_OUTPUT_FILE_PATH = Path("/output/metrics.json")
DEFAULT_GROUND_TRUTH_PATH=Path('/opt/evaluation/ground-truth')
DEFAULT_INPUT_PATH=Path('/input')

class Acrobat2022():
    def evaluate(self):
        
        #load ground truth
        try:
            gtpath=Path.joinpath(DEFAULT_GROUND_TRUTH_PATH,Path('df_points_val_container.csv'))
            self._GT = pd.read_csv(gtpath)
        except Exception as e:            
            logger.warning("Could not load ground truth.")
            stderr.write("Error 1")
            sysexit("1")            
        
        #for now only deal with CSV no zips no pickles no tsv, simply csv. 
        try:
            oneinput=list(DEFAULT_INPUT_PATH.glob("**/*.csv"))
            if len(oneinput)!=1:
                logger.warning("Only one csv is expected, got "+str(len(oneinput)))
                stderr.write("Error 2")
                sysexit("2")
            else:
                self._input=pd.read_csv(oneinput[0])                
                
        except Exception as e:
            logger.warning("Error while loading input")
         
        #If input has nans, drop them   
        self._input = self._input.dropna()

        rex=r'he_([0-9]*)_x'
        annotators=[re.search(rex, c).group(1) for c in self._GT.columns if re.search(rex, c)]       
        
        #During merge with outer we will have the correct amount of points but missing will be nans
        df_merge = pd.merge(self._GT, self._input.add_suffix('_pred'), left_on='point_id',right_on='point_id_pred', how="outer")
        
        #So we fill them with the default
        df_merge["he_x_pred"] = df_merge["he_x_pred"].fillna(df_merge["ihc_x"])
        df_merge["he_y_pred"] = df_merge["he_y_pred"].fillna(df_merge["ihc_y"])
        
        #If point i bigger than the image make it the max        
        df_merge["he_x_pred"] = df_merge["he_x_pred"].mask(df_merge["he_x_pred"]>df_merge["max_he_x"],df_merge["max_he_x"]) 
        df_merge["he_y_pred"] = df_merge["he_y_pred"].mask(df_merge["he_y_pred"]>df_merge["max_he_y"],df_merge["max_he_y"]) 
        
        #or if somehow it's less than 0 make 0
        df_merge["he_x_pred"] = df_merge["he_x_pred"].mask(df_merge["he_x_pred"]<0,0) 
        df_merge["he_y_pred"] = df_merge["he_y_pred"].mask(df_merge["he_y_pred"]<0,0) 
        
        #get annotator columns given the annotators that exist
        colsa=sum([["he_"+a+"_x","he_"+a+"_y"] for a in annotators],[])     

        #if point doesnt have both annotators (this is never the case but I leave this here as it was in the previous code)       
        df_merge.dropna(subset=colsa,inplace=True)
        
        #compute distances to each annotator and then give the save the mean of those two distances.
        meancols=[]
        for a in annotators:
            meancol="mean_d_"+a
            df_merge[meancol]=np.sqrt( (df_merge["he_"+a+"_x"]-df_merge["he_x_pred"])**2+(df_merge["he_"+a+"_y"]-df_merge["he_y_pred"])**2 )  
            meancols.append(meancol)
        df_merge["mean_mean_d_point"]=df_merge[meancols].sum(axis=1)/len(annotators)
        
        #aggregate per image and get 90th percentiles and their stats
        cols_to_rename = {'<lambda_0>':'p90','<lambda_1>':'mean'}
        df_agg=df_merge[["anon_id","mean_mean_d_point"]].groupby(by="anon_id").agg([lambda x: np.percentile(x,q=90),lambda x: np.mean(x)]).rename(columns=cols_to_rename).reset_index()
        
        df_agg.columns=df_agg.columns.droplevel(0)
        df_agg.columns=['anon_id', 'p90', 'mean']
        
        #stats
        p90s=df_agg["p90"].values
        medianp90s=np.median(p90s)
        stdp90s=np.std(p90s)
        twostdp90s=2*np.std(p90s)
        meanp90s=np.mean(p90s)
        variancep90s=np.var(p90s)
        meanofmeans=np.mean(df_agg["mean"].values)
        
        result={
            "medianp90s":medianp90s,
            "meanp90s":meanp90s,
            "variancep90s":variancep90s,
            "meanofmeans":meanofmeans,
            "stdp90s":stdp90s,
            "twostdp90s":twostdp90s
        }
        
        for i, row in df_agg.iterrows():
            result["p90-"+str(i)]=row["p90"]

        #save json file the same way as grand challenge shows
        with open(DEFAULT_EVALUATION_OUTPUT_FILE_PATH, "w") as f:
            f.write(json.dumps(result))
            
        #test are passing locally on jun 7 2022.

if __name__ == "__main__":
    Acrobat2022().evaluate()
