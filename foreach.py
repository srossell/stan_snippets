"""
foreach loop example.

Note that I'm using pystan. I noticed that Stan accepts "fixed_param" rather
than "Fixed_param", and "iter" instead of "num_samples". I have not
investigated these differences thoroughly though.
"""

import pystan

################################################################################
# INPUTS

stan_str = """
functions {
}
data {
    int<lower=1> N;
    real my_vec[N];
}
transformed data {
}
parameters {
}
transformed parameters {
}
model {
}
generated quantities {
    real out_vec[N];
    for (n in my_vec) {
        print(n);
    }
}
"""

my_vec = [2,3,5,7,11]

datadict = {
        'my_vec': my_vec,
        'N': len(my_vec)
        }

################################################################################
# STATEMENTS

# compile
sm = pystan.StanModel(model_code=stan_str)

# run
res = sm.sampling(
        data=datadict,
        algorithm='Fixed_param',
        iter=1,
        chains=1
        )

