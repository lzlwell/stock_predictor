04/21/2017 v5: update by Zhenglu:
* Add l1 or l2 norm in predict_dpi, but still using l2 for all
* Tune the hyperparameters, the current ones are good! Model is very sensitive to the models
* Add inv_diff_price()

Conclusion: The model seems working (at least on training set). According to Yicheng, the code
should be bug-free now. The model seems not be able predict very well (half-half chance). I suggest
we can move forward based on the current version.
