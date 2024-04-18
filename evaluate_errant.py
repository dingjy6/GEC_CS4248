import subprocess
import pandas as pd
from datetime import datetime

### Evaluate performance using Errant Scorer
# takes three separate csv files or three pandas dataframes/series
def get_errant_result(inputs, labels, preds, is_csv_filename = False, save_score = True, model_and_data_name = ''):
    if not is_csv_filename:
        # save to csv, ignoring row names
        inputs.to_csv('errant_inputs.csv', header = False)
        labels.to_csv('errant_labels.csv', header = False)
        preds.to_csv('errant_preds.csv', header = False)
    
        inputs_csv = 'errant_inputs.csv'
        labels_csv = 'errant_labels.csv'
        preds_csv = 'errant_preds.csv'

    else:
        inputs_csv = inputs
        labels_csv = labels
        preds_csv = preds

    create_ref_m2 = 'errant_parallel -orig ' + inputs_csv + ' -cor ' + labels_csv + ' -out true_labs_val.m2'
    subprocess.run(create_ref_m2, capture_output=True, text=True, input=None, check=True, shell=True)
    
    create_pred_m2 = 'errant_parallel -orig ' + inputs_csv + ' -cor ' + preds_csv + ' -out pred_labs_val.m2'
    subprocess.run(create_pred_m2, capture_output=True, text=True, input=None, check=True, shell=True)
    
    generate_score = 'errant_compare -hyp pred_labs_val.m2 -ref true_labs_val.m2'  # here it should not have the -ds (span-based detection) flag, default is already span-based correction
    result = subprocess.run(generate_score, capture_output=True, text=True, input=None, check=True, shell=True)
    print(result.stdout)

    if save_score:
        # Get the current date and time
        current_time = datetime.now()
        # Convert the current time to a string
        current_time_str = current_time.strftime("%Y-%m-%d_%H:%M:%S")
        save_name = './errant_res/errant_eval_' + model_and_data_name + current_time_str + '.txt'
        with open(save_name, 'w') as f:
            # Write the result to the file
            f.write(result.stdout)

    return result.stdout



# validate = pd.read_csv('t5small_50epochs_val_preds.csv') 
test = pd.read_csv('wi+locness/test_preds.csv')
res = get_errant_result(test['incorrect'], test['correct'], test['preds'], False, True, 'testset_errant')