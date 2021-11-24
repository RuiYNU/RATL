# RATL
This code is the implementation of the paper "A Relationship-Aligned Transfer Learning Algorithm for Time Series Forecasting".
Instruction on the code:

# Instruction for the code:
## Training base encoder and regressor
Once the data preparation is completed, we can train the base encoder and regressor for it.
python dataset:\train.py --lr1 0.1 --train_epochs1 200 --window 1 --neg_samples 10 --compared length None --compute_linear True
## Transfer
Once the source and target encoders and regressors are trained, we can implement the transfer phase. Separately run stage_1 and stage_2 in RATL.py
python transfer:\RATL.py --mode_1 False --train_epochs2 1000 --compute_2 True --mode_2 True --encode_pred_num 1 --encode_window 56 --test_pred_term 56
## loda model
After traning and transfer phases, we can load the saved models to predict the test data
python transfer:\get_encoder_linear.py
##
We run this code on cpu, you can change it according to the configuration.

# References
## 
In the causal_cnn.py, we implement the causal CNNs on basis of https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
## 
In the clustering_loss, we borrow idea from "Deep Temporal Clustering : Fully Unsupervised Learning of Time-Domain Features": http://arxiv.org/abs/1802.01059
## 
To deal with series with varying lengths, we borrow idea from "Unsupervised Scalable Representation Learning for Multivariate Time Series":https://proceedings.neurips.cc/paper/2019/hash/53c6de78244e9f528eb3e1cda69699bb-Abstract.html
