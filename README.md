# Adam Optimization Using Raw Google Sheets
## Intro
I wanted to test my skills to see if I could transfer an Adam Optimization code I made in a previous Google sheet into just formulas. I started working on this because I wanted to learn how to use the lambda function more and it pushed my limits to what I could do just using my brain. I worked very hard for a few days on this, and even though it may not be super fast (because it uses Google Sheets), I learned many new concepts that I had to use efficiently. It is possible to turn it all into just a single formula, however, it would not be practical and readable.

## What is Adam Optimization?
### Basics
The base function of Adam Optimization is Gradient Descent. Gradient Descent is one of the best ways to train A.I. models because it uses calculus to iteratively narrow down the best possible variables to get whatever answer is desired (especially for small values). Imagine a ball on a bumpy landscape, and depending on where the ball starts, it will roll to the deepest pit unless it gets stuck in a seemingly deep pit. The bumpy landscape is essentially the error of the calculated result, and the goal is to minimize as much error as possible. Gradient Descent would get stuck on an answer that would never improve after a certain point because it got stuck on a false answer. Adam Optimization takes Gradient Descent and adds a few new features using more calculus to get the closest answer faster and more accurately. I also have added more features that make the process more efficient.

### Technical
To do Adam Optimization, take the partial derivative of every variable, like Gradient Descent, to calculate the amount needed to nudge each variable to get a smaller error. Adam Optimization uses two new variables called momentum and velocity. Momentum and velocity can calculate the average direction for each variable to accelerate the current calculations. Adam and Gradient Descent uses a learning rate to scale how much each variable changes. A feature I added to make things super efficient is to make the learning rate smaller as the answer gets closer and closer. After a certain amount of iterations, the learning rate restarts back up and momentum and velocity can jump the calculation forward to find a closer answer even faster without that feature. After a certain point, the momentum and velocity need to be reset only after a ridiculous amount of iterations so I never use that feature in the Google sheet version.

## How I Approached This
1. I need some way to iterate functions.
2. I need to do certain array calculations without arrayformula.
3. I need to save previous parameters like variables to track certain values and use them iteratively across different formulas.
4. I need to make this as efficient as possible for Google Sheets.

I was thinking about trying this for a while, but I couldn't solve mainly the 2nd option. After I learned about the lambda function, though, everything changed. Lambda allows running sums and array calculations without using the previous functions I was stuck with. I solved the 1st option using the 'named functions' feature they recently added to Google Sheets mixed with the lambda function. The 3rd option is solved by saving everything as a seperateable string but there is a very complicated way I had to go about this. I had to make separate functions that could split the strings into arrays and use different lambda functions based on how it was formatted. Finally, the 4th option was included after I made the calculator for the second time using all the new information I learned. The key to making this run as efficiently as possible is to use the least lambda and array calculations possible and combine calculations if possible.

## All the Functions
I will list all the functions I used to explain how I did this and how everything works together to make Adam Optimization in Google Sheets.

1. SLICE(array, delimiter)
   - I needed arrays to be stored as strings to process everything equally. In Google Sheets, certain arrays would be treated differently depending on what formulas I used so I needed a standard way to finalize an array when the formula processed it. Also, I could process array strings much more easily since I could separate all the columns into 2d arrays.
- array: Any array string.
- delimiter: The string that separates each value.
```
=transpose(split(array,delimiter))
```

2. LENGTH(array, delimiter)
  - This is a quick way for me to determine the length of an array string without having to type the definition of the function every time.
- array: Any array string.
- delimiter: The string that separates each value.
```
=counta(split(array,delimiter))
```

3. BASEFUNC(input, params)
   - This function is anything that has an output based on the input and parameters the function needs. The example I used is a function to sort a large list of data with many parameters based on certain weightings the user chooses.
- input: The variables the user defines for each output.
- params: The variables that are being calculated in the end. Start with simple parameters and the calculator will use those to find the answer.
```
=byrow(input,lambda(k,let(size,LENGTH(params,","),param,SLICE(params,","),product(arrayformula(SLICE(k,",")+abs(array_constrain(param,size-2,1))))*index(param,size-1)+index(param,size))))
```

4. ADDTOALL(array, add)
   - To find the partial derivative of every variable, I needed to add a small number to every variable and calculate the result. This function adds any number to every value in the array.
- array: Any array string.
- add: Any number to add (or subtract).
```
=let(size,LENGTH(array,","),arrayformula(split(SLICE(rept(array&"|",size),"|"),",")+n(sequence(size,1)=sequence(1,size))*add))
```

5. LOSS(input, output, params)
    - The function that summarizes the total loss based on all the variables. The goal is to make this as close to zero as possible.
- input: The variables the user defines for each output.
- output: The final output the user wants.
- params: The variables that are being calculated.
```
=sum(arrayformula((BASEFUNC(input,params)-output)^2))
```

7. GRADIENT(input, output, params, prec)
   - The partial derivative of every variable to determine which direction each parameter needs to go for the smallest error. It tweaks each parameter slightly and calculates the loss to find how to lower the total loss.
- input: The variables the user defines for each output.
- output: The final output the user wants.
- params: The variables that are being calculated.
- prec: A small number used for calculating the partial derivatives AKA the precision.
```
=arrayformula(let(in,ADDTOALL(params,prec),out,ADDTOALL(params,-prec),map(sequence(LENGTH(params,",")),lambda(a,LOSS(input,output,join(",",index(in,a)))-LOSS(input,output,join(",",index(out,a))))))/(2*prec))
```

7. MOMENTUM(beta_1, momentumprev, gradients)
8. VELOCITY(beta_2, velocityprev, gradients)
