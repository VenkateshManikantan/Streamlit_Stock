# Reddit date time is converted into readable date time 

import pandas as pd

import praw
from praw.models import MoreComments

import datetime as dt
import reddit_config as r_cnf
import streamlit as st

    
    
def get_date(created):
        return dt.datetime.fromtimestamp(created)
    
    
    #Function is used to go into induvidual posts and extract the entire comment tree 
    
def extract_comm_tree_to_df_ramp(id2): #Input a ID of string; returns a df of 
        import reddit_config as r_cnf # setup your config page - for username and password for your respective reddit account
        #Setting up a reddit model
        try:
            reddit = praw.Reddit(client_id='AVu7k513AHb_DSBUp3GPPg',\
                         client_secret='YtLs8QTniQRUZHxjCL1_uUsdvMPiyA', \
                         user_agent='CABD',\
                         username= r_cnf.reddit['accessCode'] ,\
                         password= r_cnf.reddit['secretCode'] )
        except:
            print("Error in accessing redit env")
        
        post = reddit.submission(id=id2)
        Subreddit_com_dict = {
                    "score":[],\
                    "id":[],\
                    "created": [],\
                    "com_body":[],\
                    "comm_tier1":[],\
                    "comm_tier2":[]
                    }
        post.comments.replace_more(limit=0)
        comments = post.comments.list()
    
        for comment in post.comments.list():
            if isinstance(comment, MoreComments):
                continue
            Subreddit_com_dict["score"].append(comment.score)
            Subreddit_com_dict["id"].append(comment.id)
            Subreddit_com_dict["created"].append(comment.created)
            Subreddit_com_dict["com_body"].append(comment.body)
            for reply in comment.replies:
                if isinstance(reply, MoreComments):
                            continue
                Subreddit_com_dict["comm_tier1"].append(reply.body)
                for reply2 in reply.replies:
                        if isinstance(reply2, MoreComments):
                            continue
                        Subreddit_com_dict["comm_tier2"].append(reply2.body)
    
        Subreddit_com_data = pd.DataFrame.from_dict(Subreddit_com_dict, orient='index').transpose()
        _timestamp = Subreddit_com_data["created"].apply(get_date)
        Subreddit_com_data = Subreddit_com_data.assign(timestamp = _timestamp)
        Subreddit_com_data_1 = Subreddit_com_data.drop(['created'], axis=1)
        return Subreddit_com_data_1 
    
    
    #Function calls the comment tree extraction function in each of the Posts inside a subreddit:  
    
def extract_reddit_post_com_rep_ramp(subreddit_name,n): # Subreddit_name = String 
          # setup your config page - for username and password for your respective reddit account
    
        #Setting up a reddit model
        reddit = praw.Reddit(client_id='AVu7k513AHb_DSBUp3GPPg', \
                         client_secret='YtLs8QTniQRUZHxjCL1_uUsdvMPiyA', \
                         user_agent='CABD', \
                         username= r_cnf.reddit['accessCode'] , \
                         password= r_cnf.reddit['secretCode'] )
        try:
            GME_subreddit = reddit.subreddit(subreddit_name)
        except:
            print("Error in passing subreddit_name value")
        
        top_subreddit = GME_subreddit.top(limit=1000)
        Subreddit_dict = { "title":[],\
                    "score":[],\
                    "id":[],\
                    "url":[],\
                    "comms_num": [],\
                    "created": [],\
                    "body":[]}
        for submission in top_subreddit:
            Subreddit_dict["title"].append(submission.title)
            Subreddit_dict["score"].append(submission.score)
            Subreddit_dict["id"].append(submission.id)
            Subreddit_dict["url"].append(submission.url)
            Subreddit_dict["comms_num"].append(submission.num_comments)
            Subreddit_dict["created"].append(submission.created)
            Subreddit_dict["body"].append(submission.selftext)
        
        Subreddit_data = pd.DataFrame(Subreddit_dict)
        Subreddit_top_com_id = { "id":[] }
        Top_comm_posts = Subreddit_data['comms_num'].nlargest(n=n)
        for index in Top_comm_posts.index:
               Subreddit_top_com_id["id"].append(Subreddit_data.iloc[index]['id'])
        
        def get_date(created):
            return dt.datetime.fromtimestamp(created)
    
        _timestamp = Subreddit_data["created"].apply(get_date)
        Subreddit_data = Subreddit_data.assign(timestamp = _timestamp)
        Subreddit_data = Subreddit_data.drop(['created'], axis=1)
        # Top comment containeing reddit post's ID have been obtained 
        # Now to extract the 2 tier comment tree of these posts
        All_Data_Com = pd.DataFrame([])
        my_bar = st.progress(0)

        for I in range(n):
            print("Current post bieng scraped is %(post)d" % {"post":I})
            
            i = I
            q = (i+1)/n
            percent = q
            my_bar.progress(percent)
            
            Data = extract_comm_tree_to_df_ramp(Subreddit_data['id'][I])
    
            All_Data_Com = All_Data_Com.append(Data)
        
        
        return Subreddit_data,All_Data_Com;    