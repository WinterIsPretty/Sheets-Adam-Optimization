function mything(params, input) {
  let product = 1;
  for (let i = 0; i < params.length-2; i++){
    product *= Math.abs(parseFloat(params[i])) + parseFloat(input[i]);
  }
  return product*Math.abs(params[params.length-1])+parseFloat(params[params.length-2]);
}

function gradDesc(func, input, output, paramsGuess, learnRateStart = 0.01, iterations = 1000, refresh = 1000, resets = 5000, prec = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 0.00000001){
  function is2Array(params) {
    // First, check if params itself is an array
    if (!Array.isArray(params)) {
        return false;
    }

    // Then, check if every element in params is also an array
    return params.every(element => Array.isArray(element));
  }

  if (!Array.isArray(paramsGuess)) paramsGuess = [paramsGuess];
  if (!Array.isArray(output)) output = [output];
  if (!is2Array(input)) input = [input];

  let params = [...paramsGuess], prevParams = [...paramsGuess], learnRate = learnRateStart, gradient, realnum = 0, res = Array();
  let momentum = Array(paramsGuess.length).fill(0), m_prev = Array(paramsGuess.length).fill(0), m_corr = Array(paramsGuess.length), velocity = Array(paramsGuess.length).fill(0), v_prev = Array(paramsGuess.length).fill(0), v_corr = Array(paramsGuess.length), decaynum = 1;
  
  function loss(func, input, output, params){
    let loss = 0;
    for (let i = 0; i < output.length; i++){
      loss += Math.pow(func(params, input[i])-parseFloat(output[i]),2);
    }
    return loss;
  }

  function replIndex(array, index, newValue) {
    let res = [...array];
    res[index] = newValue;
    return res;
  }

  function calcGradient(func, input, output, params, index, prec){
    return (loss(func, input, output, replIndex(params,index,parseFloat(params[index])+prec)) - loss(func, input, output, replIndex(params,index,params[index]-prec))) / (2 * prec);
  }

  let currentLoss = loss(func, input, output, params);
  let prevLoss = currentLoss;
  
  currentLoss = loss(func, input, output, params);
  res.push([currentLoss,...params]);

  //Actually does the function
  for (let i = 0; i < iterations; i++){

    for (let k = 0; k < params.length; k++){
      gradient = calcGradient(func, input, output, params, k, prec)

      momentum[k] = beta_1 * momentum[k] + (1 - beta_1) * gradient;
      velocity[k] = beta_2 * velocity[k] + (1 - beta_2) * Math.pow(gradient, 2);

      m_corr[k] = momentum[k] / (1 - Math.pow(beta_1, decaynum));
      v_corr[k] = velocity[k] / (1 - Math.pow(beta_2, decaynum));
    }
    for (let k = 0; k < params.length; k++) params[k] = params[k] - (learnRate / (Math.sqrt(v_corr[k]) + epsilon))*m_corr[k];

    currentLoss = loss(func, input, output, params);
    
    if (Number.isNaN(currentLoss) || currentLoss >= prevLoss){
      //res.push([currentLoss,...params]);
      decaynum = 1;
      momentum = m_prev;
      velocity = v_prev;
      learnRate -= learnRate/10;
      params = [...prevParams];
      continue;
    }

    decaynum++;
    if(realnum%parseInt(refresh) == 0) {
      learnRate = learnRateStart;
      decaynum = 1;
    }
    if(realnum%parseInt(resets) == 0) {
      momentum = Array(paramsGuess.length).fill(0);
      velocity = Array(paramsGuess.length).fill(0);
      learnRate = learnRateStart;
      decaynum = 1;
    }

    m_prev = momentum;
    v_prev = velocity;
    prevLoss = currentLoss;
    prevParams = [...params];
    realnum++;

    //res.push([prevLoss,...params]);
  }

  return [prevLoss,...prevParams];
}

function calcVariables(input, output, paramsGuess, learnRateStart = 0.01, iterations = 1000, refresh = 10, resets = 1, prec = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 0.00000001){
  
  return gradDesc(mything, input, output, paramsGuess, learnRateStart, iterations, refresh, resets, prec, beta_1, beta_2, epsilon);
}
