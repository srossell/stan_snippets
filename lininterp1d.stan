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
    for (d in abs(deltas)) {
        if (minDeltas == d){
            break;
        }
        else{
        i += 1;
        }
    }
    if ((i + 1) >= K){
        i = K -1;
    }
    if ((deltas[i] <= 0) && (i >=2)){
        i -=1;
    }
    real x1 = x_colvec[i];
    real x2 = x_colvec[i + 1];
    real y1 = y_colvec[i];
    real y2 = y_colvec[i + 1];
    m = ((y2 - y1)/(x2 - x1));
    y = (m * (x - x1)) + y1;
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
