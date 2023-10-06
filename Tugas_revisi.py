import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    sistem_operasi = os.name

    match sistem_operasi:
        case "posix": os.system("clear")
        case "nt": os.system("cls")

# Function for Newton-Raphson
def newton_raphson(initial_guess, tolerance, max_iterations, plot=True):
    x_val = initial_guess
    iteration = 0
    
    iterations = []
    x_values = []
    f_values = []
    
    print("Iterasi\t   x\t\t   f(x)")
    
    while abs(f(x_val)) > tolerance and iteration < max_iterations:
        iterations.append(iteration)
        x_values.append(x_val)
        f_values.append(f(x_val))
        
        print("{:5d}\t{:10.6f}\t{:10.6f}".format(iteration, x_val, f(x_val)))
        
        df_x_val = df(x_val)
        if df_x_val == 0:
            raise Exception("Turunan fungsi adalah nol, metode Newton-Raphson tidak konvergen.")
        
        x_val = x_val - f(x_val) / df_x_val
        iteration += 1
    
    if abs(f(x_val)) <= tolerance:
        iterations.append(iteration)
        x_values.append(x_val)
        f_values.append(f(x_val))
        
        print("{:5d}\t{:10.6f}\t{:10.6f}".format(iteration, x_val, f(x_val)))
        
        if plot:
            # Plot the user's function
            x_vals = np.linspace(x_val - 1, x_val + 1, 400)
            y_vals = [f(val) for val in x_vals]
            
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, label='f(x)')
            plt.scatter(x_values, f_values, color='red', marker='o', label='Newton-Raphson Iterations')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.grid(True)
            plt.title('Function and Newton-Raphson Iterations')
            plt.show()
            
        return x_val
    else:
            raise Exception("Metode Newton-Raphson tidak konvergen")
        
# Function for Modified Newton-Raphson
def newton_raphson_modifikasi(initial_guess, tolerance, max_iterations, plot=True):
    x_val = initial_guess
    iteration = 0
    
    iterations = []
    x_values = []
    f_values = []

    print("Iterasi\t   x\t\t   f(x)")

    while abs(f(x_val)) > tolerance and iteration < max_iterations:
        iterations.append(iteration)
        x_values.append(x_val)
        f_values.append(f(x_val))

        print("{:5d}\t{:10.6f}\t{:10.6f}".format(iteration, x_val, f(x_val)))

        penyebut = df(x_val)**2 - f(x_val)*ddf(x_val)

        if penyebut == 0:
            raise Exception("penyebut adalah nol, metode Newton-Raphson modifikasi tidak konvergen.")

        x_val = x_val - f(x_val)*df(x_val) / penyebut
        iteration += 1

    if abs(f(x_val)) <= tolerance:
        iterations.append(iteration)
        x_values.append(x_val)
        f_values.append(f(x_val))

        print("{:5d}\t{:10.6f}\t{:10.6f}".format(iteration, x_val, f(x_val)))

        if plot:
            # Plot the user's function
            x_vals = np.linspace(x_val - 1, x_val + 1, 400)
            y_vals = [f(val) for val in x_vals]

            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, label='f(x)')
            plt.scatter(x_values, f_values, color='red', marker='o', label='Modified Newton-Raphson Iterations')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.grid(True)
            plt.title('Function and Modified Newton-Raphson Iterations')
            plt.show()

        return x_val
    else:
        raise Exception("Metode Newton-Raphson Modifikasi tidak konvergen")

# Function for Secant
def secant_method(initial_guess1, initial_guess2, tolerance, max_iterations, plot=True):
    x0 = initial_guess1
    x1 = initial_guess2
    iteration = 0

    iterations = []
    x_values = []
    f_values = []

    print("Iteration\t   x\t\t   f(x)")

    while abs(f(x1)) > tolerance and iteration < max_iterations:
        iterations.append(iteration)
        x_values.append(x1)
        f_values.append(f(x1))

        print("{:5d}\t{:10.6f}\t{:10.6f}".format(iteration, x1, f(x1)))

        # Compute the next approximation using the Secant method
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))

        x0 = x1
        x1 = x2
        iteration += 1

    if abs(f(x1)) <= tolerance:
        iterations.append(iteration)
        x_values.append(x1)
        f_values.append(f(x1))

        print("{:5d}\t{:10.6f}\t{:10.6f}".format(iteration, x1, f(x1)))

        if plot:
            # Plot the function and iterations
            x_vals = np.linspace(x1 - 1, x1 + 1, 400)
            y_vals = [f(val) for val in x_vals]

            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, label='f(x)')
            plt.scatter(x_values, f_values, color='red', marker='o', label='Iterations')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.grid(True)
            plt.title('Function and Secant Iterations')
            plt.show()

        return x1
    else:
        raise Exception("Secant method did not converge")



pilihan = input("1. Non-Linear Equation\n3. Multiple Root\n4. Linear System\nPilih! ")

if pilihan == "1":
    print("persamaan non-linear")

    # Define symbolic variable
    x = sp.symbols('x')

    # Get the mathematical function as input from the user
    function_input = input("Enter a mathematical function in terms of 'x': ")

    try:
        # Parse the user's input into a symbolic expression
        user_function = sp.sympify(function_input)

        # Print the parsed function
        print("Parsed function:", user_function)

        print("Select a method:")
        print("1. Newton-Raphson")
        print("2. Secant")

        method_choice = input("Enter the number of the method you want to use: ")

        if method_choice == "1":
            print("Newton-Raphson")
            # Define the derivative of the user's function
            user_derivative = sp.diff(user_function, x)

            # Print the derivative
            print("Derivative of the function:", user_derivative)

            # Define a numerical function from the parsed expression
            f = sp.lambdify(x, user_function, 'numpy')

            # Define a numerical derivative function from the parsed expression
            df = sp.lambdify(x, user_derivative, 'numpy')

             # Tebakan awal
            initial_guess = float(input("Enter an initial guess: "))

            # Input tolerance from the user
            tolerance = float(input("Enter tolerance: "))

            # Input maximum number of iterations from the user
            max_iterations = int(input("Enter the maximum number of iterations: "))

            # Panggil metode Newton-Raphson
            root = newton_raphson(initial_guess, tolerance, max_iterations)

            print("\nRoot: {:.6f}".format(root))

        elif  method_choice == "2":
            print("Secant")

            # Define a numerical function from the parsed expression
            f = sp.lambdify(x, user_function, 'numpy')

             # Tebakan awal
            initial_guess1 = float(input("Enter the first initial guess: "))
            initial_guess2 = float(input("Enter the second initial guess: "))

            # Input tolerance from the user
            tolerance = float(input("Enter tolerance: "))

            # Input maximum number of iterations from the user
            max_iterations = int(input("Enter the maximum number of iterations: "))

            # Call the Secant method
            root = secant_method(initial_guess1, initial_guess2, tolerance, max_iterations)

            print("\nRoot: {:.6f}".format(root))

        else:
            print("salah")
    
    except sp.SympifyError:
        print("Invalid input. Please enter a valid mathematical expression.")

elif pilihan == "2":
    print("persamaan linear")

elif pilihan == "3":
    print("Multiple Roots")
    print("Modified Newton Raphson")

    # Define symbolic variable
    x = sp.symbols('x')

    # Get the mathematical function as input from the user
    function_input = input("Enter a mathematical function in terms of 'x': ")

    try:
        # Parse the user's input into a symbolic expression
        user_function = sp.sympify(function_input)

        # Print the parsed function
        print("Parsed function:", user_function)

        # Define the derivative of the user's function
        user_derivative = sp.diff(user_function, x)

        # Print the derivative
        print("Derivative of the function:", user_derivative)

        #second derivative
        second_derrivative = sp.diff(user_derivative, x)

         # Print the 2nd derivative
        print("2nd Derivative of the function:", second_derrivative)

        # Define a numerical function from the parsed expression
        f = sp.lambdify(x, user_function, 'numpy')

        # Define a numerical derivative function from the parsed expression
        df = sp.lambdify(x, user_derivative, 'numpy')

        # Define a numerical derivative function from the parsed expression
        ddf = sp.lambdify(x, second_derrivative, 'numpy')

        # Tebakan awal
        initial_guess = float(input("Enter an initial guess: "))

        # Input tolerance from the user
        tolerance = float(input("Enter tolerance: "))

        # Input maximum number of iterations from the user
        max_iterations = int(input("Enter the maximum number of iterations: "))

        # Panggil metode Newton-Raphson
        root = newton_raphson_modifikasi(initial_guess, tolerance, max_iterations)

        print("\nRoot: {:.6f}".format(root))

    except sp.SympifyError:
        print("Invalid input. Please enter a valid mathematical expression.")

elif pilihan == "4":
    print("sistem linear")

    # Input matrix A and vector b from the user
    n = int(input("Enter the size of the matrix A (n x n): "))
    A = np.zeros((n, n))
    b = np.zeros(n)

    print("Enter the elements of matrix A:")
    for i in range(n):
        A[i, :] = input(f"Row {i + 1} (space-separated values): ").split()
        b[i] = float(input(f"Enter the corresponding value of b[{i}]: "))

    x = np.zeros(n)
    maxerr = float(input("Enter the maximum error: "))
    err1 = np.inf
    itr = 0
    y = []

    while np.all(err1 > maxerr):
        x_old = np.copy(x)
        for i in range(n):
            total = 0
            for j in range(i):
                total += A[i, j] * x[j]
            for j in range(i + 1, n):
                total += A[i, j] * x_old[j]
            x[i] = (1 / A[i, i]) * (b[i] - total)
            print(f'Iteration {itr}, x[{i}] = {x[i]}')  # Display the current iteration
        itr += 1
        y.append(x.copy())
        err1 = np.abs(x_old - x)

    print("Number of iterations:", itr)
    print("Final solution:", x)

else:
    print("SALAH!")




