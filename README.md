# mulfi_model

### Multi-fidelity Model

This model is developed as multi-fidelty model. Phthon 2.7 is used for the language.

More efficient than Kriging model, while maintains the same accuracy.

Suitable for fast robust optimization and prediction.

---
cokriging.py is the main code for multi-fidelity model;

krige.py is the code for Kriging model, which is also useful for multi-fidelity model;

matrixops.py is the code for matrix operations, which is basis for co-kriging.py as well as krige.py;

LHS.py is the latin hypercubic sampling mehtod code;
