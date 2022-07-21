import pickle as pk
import numpy as np
model=pk.load(open('modelprediction/model_save','rb'))

def processing(d):
    l=['Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Source_Chennai', 'Source_Delhi',
       'Source_Kolkata', 'Source_Mumbai', 'Airline_Air India', 'Airline_GoAir',
       'Airline_IndiGo', 'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Total_Stops', 'Duration_Hours', 'Duration_Min', 'Day_Of_Journey',
       'Month_Of_Journey', 'Dep_hour', 'Dep_min', 'Arrival_hour',
       'Arrival_min']
    destination={
        'Cochin':0,
        'Delhi':0,
        'Hyderabad':0,
        'Kolkata':0
    }
    
    destination[d['destination']]=1
    source={
        'Channai':0,
        'Delhi':0,
        'Kolkata':0,
        'Mumbai':0
    }
    
    source[d['source']]=1
    airlines_dict={
    'Air India':0,
    'GoAir':0,
    'IndiGo':0,
    'Jet Airways':0,
    'Jet Airways Business':0,
    'Multiple carriers':0,
    'Multiple carriers Premium economy':0,
    'SpiceJet':0,
    'Trujet':0,
    'Vistara':0,
    'Vistara Premium economy':0
    }
    airlines_dict[d['company']]=1
    no_of_stops=int(d['stops'])
    day=int(d['datej'][-2:])
    month=int(d['datej'][-5:-3])
    dep_h=int(d['departure'][0:2])
    dep_m=int(d['departure'][3:])
    arr_h=int(d['arrival'][:2])
    arr_m=int(d['arrival'][3:])
    dur_h=abs(dep_h-arr_h)
    dur_m=abs(dep_m-arr_m)
    
    def make_list(destination,source,airlines_dict,extra):
        l=[]
        for i in destination.keys():
            l.append(destination[i])
        for i in source.keys():
            l.append(source[i])
        for i in airlines_dict.key():
            l.append(airlines_dict[i])
        for i in extra:
            l.append(i)
        return l

    features_list=make_list(destination,source,airlines_dict,[no_of_stops,dur_h,dur_m,day,month,dep_h,dep_m,arr_h,arr_m])
    l=np.array([[features_list]])
  
    actual= (round(model.predict(l)[0],2))
    
    def calc_for_every_flight(destination,source,airlines_dict,extra,model):
        airlines={'Air India':0, 'GoAir':0,
        'IndiGo':0, 'Jet Airways':0, 
        'SpiceJet':0,
        'Trujet':0, 'Vistara':0}
        for i in airlines_dict.keys():
            airlines_dict[i]=0
        for i in airlines:
            airlines_dict[i]=1
            features_list=make_list(destination,source,airlines_dict,extra)
            l=np.array([[features_list]])
            airlines[i]=(round(model.predict(l)[0],2))
            airlines_dict[i]=0
    airlines=calc_for_every_flight(destination,source,airlines_dict,[no_of_stops,dur_h,dur_m,day,month,dep_h,dep_m,arr_h,arr_m],model)
    airlines['actual']=actual
    return airlines
    
    
    
       