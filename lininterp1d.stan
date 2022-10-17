// https://discourse.mc-stan.org/t/linear-interpolation-and-searchsorted-in-stan/13318/5
// MODIFIED. sort_indices_asc doesn't work as expected
functions{
real lininterp1d(real x, vector x_colvec, vector y_colvec){
    int K = rows(x_colvec);
    real m;
    real b;
    real y;
    int i;
    // get index of x_colvec closest to x
    vector[K] deltas = x - x_colvec;
    i = 1;
    real minDeltas;
    minDeltas = min(abs(deltas));
    for (d in deltas) {
        if (minDeltas == d){
            break;
        }
        else{
        i += 1;
        }
    }
    // Input checks
    if(x<x_colvec[1] || x>x_colvec[K]) reject("x is outside of the x_colvec grid!");
    if(rows(y_colvec) != K) reject("x_colvec and y_colvec aren't of the same size");
    // end Input checks
    real x1 = x_colvec[i];
    real x2 = x_colvec[i + 1];
    real y1 = y_colvec[i];
    real y2 = y_colvec[i + 1];
    m = ((y2 - y1)/(x2 - x1));
    b = y1 - (m * x1);
    y = (m * x) + b;
    return y;
  }

}

data {
	int<lower=1> data_dim0;  // length data vectors
    int<lower=1> x2interp_dim0;  // Length vector of vals to interpolate
	vector[x2interp_dim0] x2interp;
    # real x2interp;
	vector[data_dim0] given_x;
	vector[data_dim0] given_y;
}

generated quantities {
	vector[x2interp_dim0] yinterp;
    for (i in 1:x2interp_dim0) {
    	yinterp[i] = lininterp1d(x2interp[i], given_x, given_y);
    }
}
