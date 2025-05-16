import tensorflow as tf
import numpy as np
import csv

class val_mae(tf.keras.callbacks.Callback):
    
    def __init__(self, inp_sequence, y):
        super().__init__()
        self.inp_sequence = inp_sequence
        self.y = y

    def on_train_begin(self, logs=None):
        self.best = np.Inf
        with open('val_mae.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Epoch','Mean Absolute Error'])
        csvfile.close()
    
    def on_epoch_end(self, epoch, logs=None):
        
        mean = []
        mean_true = np.concatenate(self.y, axis=0)
        model = self.model
        
        for x in self.inp_sequence:
            mean_pred = model(x).mean().numpy().flatten()
            mean.extend(mean_pred)
                    
        mean = np.array(mean) 
        mean_error = np.mean(np.absolute(mean - mean_true))
        
        with open('val_mae.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([epoch,round(mean_error, 8)])
            csvfile.close()
            
        if mean_error < self.best:
            print("\n\nEpoch "+ str(epoch+1)+": Validation Mean Absolute Error Improved from "+ str(round(self.best,4)) + " to " + str(round(mean_error, 4)))
            self.best = mean_error
            model.save_weights('best_model_val_mae.h5')
            
        else:
            print("\n\nEpoch "+ str(epoch+1)+": Validation Mean Absolute Error did not from "+ str(round(self.best,4)))
            
            
        