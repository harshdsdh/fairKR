# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 18:38:11 2021

@author: harsh
"""
#from flask import Flask
from gensim import models
from gensim import similarities
from gensim.corpora import Dictionary
import pandas as pd
import numpy as np
import collections

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


emb_glove = KeyedVectors.load_word2vec_format('glove.txt', binary=False)

main_df = pd.read_csv('main_workshop.csv')
democrats_df1 =  pd.read_csv('democrats_workshop.csv')
republicans_df1 = pd.read_csv('republicans_workshop.csv')

main_df = main_df.dropna(subset = ['text']).reset_index(drop=True)
democrats_df1 = democrats_df1.dropna(subset = ['text']).reset_index(drop=True)
republicans_df1 = republicans_df1.dropna(subset = ['text']).reset_index(drop=True)

documents = main_df['text']
def frequency_mapping(democrats_df1, republicans_df1):
    x_democrats = ' '.join(democrats_df1['text']).split(' ')
    #print(x_democrats)
    map_democrats = collections.Counter(x_democrats)
    
    x_republicans = ' '.join(republicans_df1['text']).split(' ')
    map_republicans = collections.Counter(x_republicans)
    return map_democrats, map_republicans

map_democrats, map_republicans = frequency_mapping(democrats_df1, republicans_df1)
tfidf_vectorizer = TfidfVectorizer()
tfidf_weights_matrix = tfidf_vectorizer.fit_transform(main_df['text'])

def analyze(main_df, sim):
    democrats_suggestions=[]
    republicans_suggestions=[]
    
    for sent in sim:
        politics  = main_df.loc[sent]['politics']
        if politics==1:
            democrats_suggestions.append(sent)
        else:
            republicans_suggestions.append(sent)
    return democrats_suggestions,republicans_suggestions


def suggest_words_sim(word, map_democrats, map_republicans):
    result = emb_glove.most_similar(word)[:10]
    result_list = [key for key,dist in result]+[word]
    suggest_list_democrats = []
    suggest_list_republicans=[]

    for word in result_list:
        
        if map_democrats[word]<=map_republicans[word] and map_republicans[word]!=0:
            suggest_list_republicans.append(word)
        if map_democrats[word]>=map_republicans[word] and map_democrats[word]!=0:
            suggest_list_democrats.append(word)
    return suggest_list_democrats, suggest_list_republicans

def analyse_search_query(query, map_democrats, map_republicans):
    arr = query.split(' ')
    res= {}
    suggestions={
    }
    suggestions['words_dems']={}
    suggestions['words_repub']={}
    
    for word in arr:
        if word in emb_glove:
            res[word]= [word]
    
    for k,v in res.items():
        suggestions['words_dems'][k]  = suggest_words_sim(k,map_democrats, map_republicans)[0]
        suggestions['words_repub'][k]  = suggest_words_sim(k,map_democrats, map_republicans)[1]
        
    return(suggestions)

def find_word_embedding_rel(query):
    we_rel  = emb_glove.most_similar(query)[:10]
    we_rel_map={}
    for word, rel in we_rel:
        we_rel_map[word] = round(rel,3)
    return we_rel_map

def create_document_dict(query, main_df, democrats_sugg_orig, republicans_sugg_orig):
    doc_orig_index = democrats_sugg_orig+ republicans_sugg_orig
    doc_orig={}
    doc_orig['orig']={}
    doc_orig['orig']['query']=query
    doc_orig['orig']["res"]=[]
    doc_orig['orig']['group']=[]
    
    for i in doc_orig_index:
        doc_orig['orig']['res'].append(main_df.loc[i]["text"])
    doc_orig['orig']['group'].append(len(democrats_sugg_orig))
    doc_orig['orig']['group'].append(len(republicans_sugg_orig))
    return doc_orig
    
def create_dataset_pareto(query, democrats_sugg, republicans_sugg,i, doc_orig, main_df):
    doc_orig[i]={}
    doc_orig[i]['query']=query
    doc_orig[i]['res']=[]
    doc_index = democrats_sugg+republicans_sugg
    for ind in doc_index:
        doc_orig[i]['res'].append(main_df.loc[ind]["text"])
    doc_orig[i]['group']=[len(democrats_sugg), len(republicans_sugg)]
    return doc_orig

def link_embedding_to_search_engine(query, main_df, suggestions_dict,democrats_sugg_orig, republicans_sugg_orig, document_dict):
    score_dems={}
    score_repub={}
    i=1
    for k,v in suggestions_dict['words_dems'].items():
        for word in v:
            temp = query.replace(k,word)
            search_engine_result = cosine_search_engine(main_df, temp)
    
            democrats_sugg, republicans_sugg = analyze(main_df, search_engine_result)
            
            #score[k,word] = [len(democrats_sugg), len(republicans_sugg), temp]
            
            create_dataset_pareto(word, democrats_sugg,republicans_sugg,i, document_dict, main_df)
            i+=1
            score_dems[k,word] = [len(democrats_sugg), len(republicans_sugg), temp]
    
    for k,v in suggestions_dict['words_repub'].items():
        for word in v:
            temp = query.replace(k,word)
            search_engine_result = cosine_search_engine(main_df, temp)
    
            democrats_sugg, republicans_sugg = analyze(main_df, search_engine_result)
            
            #score[k,word] = [len(democrats_sugg), len(republicans_sugg), temp]
            
            create_dataset_pareto(word, democrats_sugg,republicans_sugg,i, document_dict, main_df)
            i+=1
            score_repub[k,word] = [len(democrats_sugg), len(republicans_sugg), temp]
    return(score_dems, score_repub)
    
def get_bias(group):
    A = group[0]
    B = group[1]
    
    return round((A-B)/(A+B),2)

def getRelevanceScores(dataobject, query):
    doc1 = dataobject['orig']['res']
    key_arr = dataobject.keys()
    keys_range=0
    for val in key_arr:
        if isinstance(val, int):
            keys_range = max(val, keys_range)
    #keys_range = list(dataobject.keys())[-1]
    #print("kr", keys_range)
    
    relevance_set={}
    cos_sim=[]
    ans_set={}
    max_recall_val=0
    max_precison_val=0
    
    orig_relevance_set={}
    orig_relevance = 0
    orig_recall=[]
    orig_precision=[]
    temp_query = query.split(' ')
    sent2 = temp_query
    orig_relevance=1.0
    
    for i in range(1,keys_range+1):
        doc = dataobject[i]['res']
        q= dataobject[i]['query']
        relevance_set[q]={}
        recall_set=[]
        precision_set=[]
        for d in doc1:
            sent1 = [w for w in d.split(' ') if w in emb_glove.vocab] 
            max_recall_val=0
            for c in doc:
                sent2=[w for w in c.split(' ') if w in emb_glove.vocab]
                if sent1 and sent2:
                    max_recall_val = max(max_recall_val,emb_glove.n_similarity(sent1, sent2))
            recall_set.append(max_recall_val)
        relevance_set[q]['recall'] = round(sum(recall_set)/len(recall_set),2)
        #print(q,relevance_set[q]['recall'])
        for c in doc:
            sent1=[w for w in c.split(' ') if w in emb_glove.vocab]
            max_precison_val=0
            for d in doc1:
                sent2 = [w for w in d.split(' ') if w in emb_glove.vocab] 
                if sent1 and sent2:
                    max_precison_val = max(max_precison_val,emb_glove.n_similarity(sent1, sent2))
            precision_set.append(max_precison_val)
        relevance_set[q]['precision'] = round(sum(precision_set)/len(precision_set),2)
        #print(q,relevance_set[q]['precision'])
        relevance_set[q]['relevance'] = round(((2*relevance_set[q]['precision']*relevance_set[q]['recall'])/(relevance_set[q]['recall']+relevance_set[q]['precision'])),2)
        
        #cos_sim.append(get_recall_sim(doc1, doc))
        relevance_set[q]['metrics'] = [relevance_set[q]['relevance'], get_bias(dataobject[i]["group"])]
        #print(q, get_bias(dataobject[i]["group"]), abs(original_bias))
        #if abs(get_bias(dataobject[i]["group"]))<=abs(original_bias):
        ans_set[q] = [relevance_set[q]['relevance'], get_bias(dataobject[i]["group"])]
    ans_set[query][0] = 1.0
    return ans_set

def make_result_set(result_arr, map1, map2):
    temp=[]
    #basic_query=query
    for i in result_arr:
        suggested_word = i[2]
        for k,v in map1.items():
            if suggested_word in v:
                temp.append([k+" -> "+suggested_word])
        for k,v in map2.items():
            if suggested_word in v:
                temp.append([k+" -> "+suggested_word])
    return temp
def build_paretofront(sorted_list,bias,maxY=True):
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] > pareto_front[-1][1] and pair not in pareto_front and abs(pair[0])<=abs(bias):
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1] and pair not in pareto_front:
                pareto_front.append(pair)
    return pareto_front

def merge_intervals(intervals):
    intervals.sort(key= lambda x:x[0])
    i=0
    n = len(intervals)
    merge=[]
    for interval in intervals:
        if not merge or merge[-1][0]!=interval[0]:
            merge.append(interval)
        else:
            merge[-1][1] = max(interval[1], merge[-1][1])
    return merge


def plot_sns_pareto_frontier(Xs, Ys, original_bias, rel_set,query,we_rel_map, maxX=True, maxY=True):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))])
    positive_bias=[]
    negative_bias=[]
    min_rel=float('inf')
    for bias,rel in sorted_list:
        if original_bias<0:
            if bias>0:
                positive_bias.append([bias, rel])
            if bias<=0:
                negative_bias.append([bias, rel])
        else:
            if bias>=0:
                positive_bias.append([bias, rel])
            if bias<0:
                negative_bias.append([bias, rel])
        min_rel = min(min_rel, rel)
    
    positive_bias = merge_intervals(positive_bias)
    negative_bias = merge_intervals(negative_bias)
    negative_bias.sort(key= lambda x:x[0], reverse=True)
    
    if len(positive_bias)>0:
        pf_positive_bias = build_paretofront(positive_bias,original_bias, True)
    else:
        pf_positive_bias=[]
    if len(negative_bias):
        pf_negative_bias  = build_paretofront(negative_bias,original_bias, True)
    else:
        pf_negative_bias=[]
    
    
    pareto_front = pf_positive_bias+pf_negative_bias
    
    for i in pareto_front:
        for k,v in rel_set.items():
            v=list(v)[::-1]
            if v==i:
                i.append(k)
                
   

    pf_X = [pair[0] for pair in pf_positive_bias if abs(pair[0])<=abs(original_bias)]
    pf_Y = [pair[1] for pair in pf_positive_bias if abs(pair[0])<=abs(original_bias)]

    
    pf_X = [pair[0] for pair in pf_negative_bias if abs(pair[0])<=abs(original_bias)]
    pf_Y = [pair[1] for pair in pf_negative_bias if abs(pair[0])<=abs(original_bias)]
    

    return pareto_front

def tf_idf(search_keys):
    search_query_weights = tfidf_vectorizer.transform([search_keys])

    return search_query_weights, tfidf_weights_matrix

def cos_similarity(search_query_weights, tfidf_weights_matrix):

    cosine_distance = cosine_similarity(search_query_weights, tfidf_weights_matrix)
    similarity_list = cosine_distance[0]
  
    return similarity_list

def most_similar(similarity_list, min_talks=1):

    most_similar= []
  
    while min_talks > 0:
        tmp_index = np.argmax(similarity_list)
        if tmp_index not in most_similar and tmp_index!=0:
            most_similar.append(tmp_index)
            similarity_list[tmp_index] = 0
        min_talks -= 1

    return most_similar
def cosine_search_engine(main_df, query):
    search_query_weights, tfidf_weights_matrix = tf_idf(query)
    l = cos_similarity(search_query_weights, tfidf_weights_matrix)
    result = most_similar(l,20)    
    
    return result
        
        

def create_docs(query='riots'):
    search_engine_result = cosine_search_engine(main_df, query)
    democrats_sugg_orig, republicans_sugg_orig = analyze(main_df, search_engine_result)
    original_bias = round((len(democrats_sugg_orig)-len(republicans_sugg_orig))/(len(democrats_sugg_orig)+len(republicans_sugg_orig)),2)
    
    
    suggest_list_democrats, suggest_list_republicans = suggest_words_sim(query, map_democrats, map_republicans)
    suggestions_embedding = analyse_search_query(query,map_democrats, map_republicans)
    map1 = suggestions_embedding['words_dems']
    map2  = suggestions_embedding['words_repub']
    we_rel_map = find_word_embedding_rel(query)
    
    document_dict = create_document_dict(query, main_df, democrats_sugg_orig, republicans_sugg_orig)
    
    score_dems, score_reps =link_embedding_to_search_engine(query, main_df, suggestions_embedding,democrats_sugg_orig, republicans_sugg_orig, document_dict)
    rel_set = getRelevanceScores(document_dict, query)
    
    bias_arr_graph=[]
    
    res_arr_graph=[]
    word_graph=[]
    for k, v in rel_set.items():
        if k in we_rel_map:
            word_graph.append(k+' '+'('+str(we_rel_map[k])+')')
        else:
            word_graph.append('query: '+k)
        res_arr_graph.append(v[0])
        bias_arr_graph.append(v[1])
    '''graph_data={}
    graph_data['tfidf word'] = word_graph
    graph_data['bias']=bias_arr_graph
    graph_data['relevance'] = res_arr_graph
    graph_data = pd.DataFrame(graph_data)
    graph_data.to_csv('graph_scatter.csv',mode='a', index=False)'''
    
    
    scores = list(rel_set.values())
    X = [i[1] for i in scores]
    Y=[i[0] for i in scores]
    
    result_arr = plot_sns_pareto_frontier(X,Y, original_bias,rel_set,query,we_rel_map, True, True)
    print(result_arr)
    resp_data = {}
    s=''
    for ind, val in enumerate(result_arr):
        resp_data[val[2]] = [{'bias': val[0], 'relevance': val[1]}]
        s+='| '+ val[2]
    print(s)
    
    return resp_data
   
create_docs()
