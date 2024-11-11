# 7_api
Create API to deploy ML model

**Fast API**: one the most common frameworks

1. First decorator loads the pickle where the model is stored 

``` python
@app.on_event("startup")
