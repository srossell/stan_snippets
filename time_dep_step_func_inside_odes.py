"""
Noiseless simulation using stan including a time-dependent stepwise function
inside a system of ordinary differential equations.

The system of odes represents a simple linear pathway:
    s -> x1 -> x2

The substrate "s" is modeled as time-dependent stepwise function. The rate of
production of "x1" is the value of "s", its rate of consumption is modeled as an
irreversible Michaelis Menten equation. "x2" is produced from "x1" and through
the same rate law (representing the same enzyme).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan



###############################################################################
# INPUTS

var_dict = {
        'y0[1]':'x1_0',
        'y0[2]': 'x2_0',
        }

stan_str = """
functions {
    real f_mm (
        real[] y
    )
    {
        real v_mm;
        v_mm = (y[1]/1) / (1 + (y[1]/1));
        return v_mm;
    }

    real subs (
        real t
    )
    {
        real s;
        if (t<2)
            s = 1;
        else if ((t >= 2) && (t < 5))
            s = exp(-t);
        else
            s = 0.2;
        return s;
    }

    real[] myodes(
        real t,
        real[] y,
        real[] p,
        real[] x_r,
        int[] x_i
        )
        {
            real dydt[2];
            dydt[1] = subs(t) - f_mm(y);
            dydt[2] = f_mm(y);
            return dydt;
        }
}

data {
    int<lower=1> T;
    real t0;
    real t_sim[T];
    real<lower=0> y0[2]; // init
    }

transformed data {
    real x_r[0];
    int x_i[0];
}

parameters {
    vector<lower=0>[2] sigma;
    real<lower=0> p[2];
}

transformed parameters {
    real y_hat[T, 2];
    y_hat = integrate_ode_rk45(myodes, y0, t0, t_sim, p, x_r, x_i);
}

model {
}

generated quantities {
}
"""
###############################################################################
# STATEMENTS

# Dictionary to pass to the stan model
t_sim = np.arange(1, 10, 0.5)
datadict = {}
datadict['t0'] = 0
datadict['t_sim'] = t_sim
datadict['T'] = len(t_sim)
datadict['y0'] = [0, 0]

# compile StanModel
sm = pystan.StanModel(model_code=stan_str)

fit = sm.sampling(
                    data=datadict,
                    iter=1,
                    chains=1,
                    n_jobs=1,
                    algorithm='Fixed_param',
                    seed = 42
                    )

summary = fit.summary()
df_summary =  pd.DataFrame(
    summary['summary'],
    columns=summary['summary_colnames'],
    index=summary['summary_rownames']
)

fit_dict= fit.extract()
fit_df = pd.DataFrame(fit_dict['y_hat'][0], index=t_sim, columns=['x1', 'x2'])

fit_df.plot(subplots=True)
plt.show()

