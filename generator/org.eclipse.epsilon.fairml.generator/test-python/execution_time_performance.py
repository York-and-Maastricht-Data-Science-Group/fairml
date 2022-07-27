import time
from datetime import datetime

all_start = time.time()

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

import tkinter
import sys 
import gc
from importlib import import_module


generated_code_path = '../test-model'
baseline_code_path = '../ibmfai360'
generated_key = 'generated'
original_key = 'original'

results = {}

results["tutorial_medical_expenditure"]  = None
results["demo_exponentiated_gradient_reduction"] = None
results["demo_disparate_impact_remover"]  = None
results["demo_meta_classifier"]  = None
results["demo_optim_preproc_adult"]  = None
results["demo_short_gerryfair_test"]  = None
results["demo_reweighing_preproc"]  = None
results["tutorial_credit_scoring"] = None
results["demo_reject_option_classification"] = None
results["demo_adversarial_debiasing"] = None
results["demo_calibrated_eqodds_postprocessing"] = None
results["demo_lfr"] = None

## multiple iteration (the ideal performance measure)
# threshold = 5 #5
# fr = 1 #1
# to = 14 #14

## single iteration (just to get through all the examples once)
threshold = 0 #5
fr = 1 #1
to = 2 #14

for key in results:
    results[key] = {generated_key: 0, original_key:0}

def meaure_execution_time(module_name):
    
    generated_code_time = []
    original_code_time = []
    
    for i in range(fr, to):
        
        ''' === generated_code_time === '''
        sys.path.insert(0, generated_code_path)    
        
        start = time.process_time()
        import_module(module_name)
        end = time.process_time()
        
        if i >= threshold: 
            generated_code_time.append(end - start)
        
        sys.path.remove(generated_code_path) 
        del sys.modules[module_name]
        gc.collect()
        
        ''' === Original === '''
        sys.path.insert(0, baseline_code_path)
        
        start = time.process_time()
        import_module(module_name)
        end = time.process_time()
        
        if i >= threshold: 
            original_code_time.append(end - start)
         
        sys.path.insert(0, baseline_code_path)
        del sys.modules[module_name]
        gc.collect()
        
    
    results[module_name][generated_key] = sum(generated_code_time) / len(generated_code_time)
    results[module_name][original_key] = sum(original_code_time) / len(original_code_time)

    del generated_code_time
    del original_code_time

def capitalise(text):
    temp = ""
    prev_c = None
    for c in text:
        if prev_c == None:
            temp = temp + c.capitalize()
        elif prev_c == "_":
            temp = temp + c.capitalize()
        elif c == "_":
            temp = temp + " "
        else:
            temp = temp + c
        
        prev_c = c
    
    return temp


for key in results:
    meaure_execution_time(key)

print("\nALFA")
print("Example\tGenerated\tOriginal")
for key in results:
    print("{}\t{:.3f}\t{:.3f}".format(capitalise(key), results[key][generated_key], results[key][original_key]))
    
    
all_end = time.time()
print("\n\n\nTotal Time: {:.3f} minute(s)".format((all_end - all_start)/60.0))