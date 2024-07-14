from flask import Flask, render_template, request
from sympy import symbols, integrate, ln, latex
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')  # Utilise un backend sans interface graphique

import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def format_result(result):
    # Remplacer "log" par "ln"
    return str(result).replace("log", "ln")

@app.route('/calculator')
def calculator():
    return render_template('calculator.html')

# Résoudre les systèmes linéaires
@app.route('/solve_linear_system', methods=['GET', 'POST'])
def solve_linear_system():
    if request.method == 'POST':
        A = sp.Matrix(eval(request.form['A']))
        b = sp.Matrix(eval(request.form['b']))
        solution = A.LUsolve(b)
        result_latex = sp.latex(solution)
        return render_template('result.html', result=result_latex)
    return render_template('solve_linear_system.html')

# Trouver les valeurs propres et les vecteurs propres
@app.route('/eigenvalues_eigenvectors', methods=['GET', 'POST'])
def eigenvalues_eigenvectors():
    if request.method == 'POST':
        A = sp.Matrix(eval(request.form['A']))
        eigenvals = A.eigenvals()
        eigenvects = A.eigenvects()
        result_latex = f"Valeurs Propres: {sp.latex(eigenvals)}, Vecteurs Propres: {sp.latex(eigenvects)}"
        return render_template('result.html', result=result_latex)
    return render_template('eigenvalues_eigenvectors.html')

# Décomposition LU
@app.route('/lu_decomposition', methods=['GET', 'POST'])
def lu_decomposition():
    if request.method == 'POST':
        A = sp.Matrix(eval(request.form['A']))
        L, U, _ = A.LUdecomposition()
        result_latex = f"L: {sp.latex(L)}, U: {sp.latex(U)}"
        return render_template('result.html', result=result_latex)
    return render_template('lu_decomposition.html')

# Trouver le min et le max d’une fonction
@app.route('/find_min_max', methods=['GET', 'POST'])
def find_min_max():
    if request.method == 'POST':
        func = request.form['func']
        a = float(request.form['a'])
        b = float(request.form['b'])
        x = sp.symbols('x')
        f = sp.sympify(func)
        deriv = sp.diff(f, x)
        critical_points = sp.solve(deriv, x)
        critical_points = [point for point in critical_points if a <= point <= b]
        critical_points.extend([a, b])
        critical_points = list(set(critical_points))
        values = [f.subs(x, point) for point in critical_points]
        min_value = min(values)
        max_value = max(values)
        result_latex = f"Min: {sp.latex(min_value)}, Max: {sp.latex(max_value)}"
        return render_template('result.html', result=result_latex)
    return render_template('find_min_max.html')

# Exemple de renvoi sur les primitives
@app.route('/calculate_integral', methods=['GET', 'POST'])
def calculate_integral():
    if request.method == 'POST':
        expression = request.form['expression']
        var = request.form['variable']
        x = symbols(var)
        integral = integrate(ln(x), x)
        result = latex(integral).replace("log", "ln")  # Remplacer log par ln
        return render_template('result.html', result=result)
    return render_template('calculate_integral.html')


# Développer et simplifier les expressions
@app.route('/simplify_expression', methods=['GET', 'POST'])
def simplify_expression():
    if request.method == 'POST':
        expression = request.form['expression']
        action = request.form['action']
        expr = sp.sympify(expression)
        
        if action == 'simplify':
            result = sp.simplify(expr)
        elif action == 'expand':
            result = sp.expand(expr)
        elif action == 'factor':
            result = sp.factor(expr)
        
        result_latex = sp.latex(result)
        return render_template('result.html', result=result_latex)
    
    return render_template('simplify_expression.html')

# Calculer des dérivées
@app.route('/calculate_derivative', methods=['GET', 'POST'])
def calculate_derivative():
    if request.method == 'POST':
        expression = request.form['expression']
        variable = request.form['variable']
        var = sp.symbols(variable)
        derivative = sp.diff(expression.replace('log', 'ln'), var)
        result_latex = sp.latex(derivative)
        return render_template('result.html', result=result_latex)
    return render_template('calculate_derivative.html')

# Développement limité (calcul de séries)
@app.route('/taylor_series', methods=['GET', 'POST'])
def taylor_series():
    if request.method == 'POST':
        expression = request.form['expression']
        variable = request.form['variable']
        point = float(request.form['point'])
        order = int(request.form['order'])
        var = sp.symbols(variable)
        taylor = sp.series(expression.replace('log', 'ln'), var, point, order).removeO()
        result_latex = sp.latex(taylor)
        return render_template('result.html', result=result_latex)
    return render_template('taylor_series.html')

@app.route('/calculate_limit', methods=['GET', 'POST'])
def calculate_limit():
    if request.method == 'POST':
        expression = request.form['expression']
        variable = request.form['variable']
        point = request.form['point']
        
        var = sp.symbols(variable)
        
        # Convertir "oo" et "-oo"
        if point == 'oo':
            point = sp.oo
        elif point == '-oo':
            point = -sp.oo
        else:
            try:
                point = float(point)  # Convertir en float
            except ValueError:
                return render_template('calculate_limit.html', error_message="Veuillez entrer une valeur valide.")
        
        limit = sp.limit(sp.sympify(expression), var, point)
        return render_template('result.html', result=limit)
    
    return render_template('calculate_limit.html')



# Tracer des graphiques
@app.route('/plot_graph', methods=['GET', 'POST'])
def plot_graph():
    if request.method == 'POST':
        expression = request.form['expression']
        variable = request.form['variable']
        a = float(request.form['a'])
        b = float(request.form['b'])
        var = sp.symbols(variable)
        func = sp.lambdify(var, sp.sympify(expression.replace('log', 'ln')), 'numpy')
        x_vals = np.linspace(a, b, 400)
        y_vals = func(x_vals)
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=expression)
        plt.xlabel(variable)
        plt.ylabel(f'f({variable})')
        plt.legend()
        plt.grid(True)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return render_template('result.html', result=f'<img src="data:image/png;base64,{plot_url}"/>')
    return render_template('plot_graph.html')


@app.route('/gram_schmidt', methods=['GET', 'POST'])
def gram_schmidt():
    if request.method == 'POST':
        vectors_data = request.form['vectors']
        try:
            vectors = [sp.Matrix([float(x) for x in vec.split()]) for vec in vectors_data.split('\n')]
            ortho_vectors = sp.gram_schmidt(vectors)
            result_latex = sp.latex(ortho_vectors)
            return render_template('result.html', result=result_latex)
        except Exception as e:
            return render_template('gram_schmidt.html', error_message=str(e))
    return render_template('gram_schmidt.html')

@app.route('/gcd_lcm_bezout', methods=['GET', 'POST'])
def gcd_lcm_bezout():
    if request.method == 'POST':
        try:
            a = int(request.form['a'])
            b = int(request.form['b'])
            gcd, x, y = sp.gcdex(a, b)
            lcm = abs(a * b) // gcd
            result_latex = f"PGCD: {sp.latex(gcd)}, PPCM: {sp.latex(lcm)}, Identité de Bézout: {sp.latex(x)}*{a} + {sp.latex(y)}*{b} = {sp.latex(gcd)}"
            return render_template('result.html', result=result_latex)
        except Exception as e:
            return render_template('gcd_lcm_bezout.html', error_message=str(e))
    return render_template('gcd_lcm_bezout.html')

@app.route('/points_critiques', methods=['GET', 'POST'])
def points_critiques():
    if request.method == 'POST':
        function_data = request.form['function']
        try:
            x, y = sp.symbols('x y')
            f = sp.sympify(function_data)
            gradient = sp.Matrix([sp.diff(f, var) for var in (x, y)])
            critical_points = sp.solve(gradient, (x, y))
            result_latex = sp.latex(critical_points)
            return render_template('result.html', result=result_latex)
        except Exception as e:
            return render_template('points_critiques.html', error_message=str(e))
    return render_template('points_critiques.html')

@app.route('/equations', methods=['GET', 'POST'])
def equations():
    if request.method == 'POST':
        equation_data = request.form['equation']
        try:
            x = sp.symbols('x')
            equation = sp.sympify(equation_data.replace('log', 'ln'))
            solutions = sp.solve(equation, x)
            result_latex = sp.latex(solutions)
            return render_template('result.html', result=result_latex)
        except Exception as e:
            return render_template('equations.html', error_message=str(e))
    return render_template('equations.html')

@app.route('/equations_differentielles', methods=['GET', 'POST'])
def equations_differentielles():
    if request.method == 'POST':
        equation_data = request.form['equation']
        try:
            x = sp.symbols('x')
            y = sp.Function('y')(x)
            equation = sp.sympify(equation_data.replace('log', 'ln'))
            solutions = sp.dsolve(equation, y)
            result_latex = sp.latex(solutions)
            return render_template('result.html', result=result_latex)
        except Exception as e:
            return render_template('equations_differentielles.html', error_message=str(e))
    return render_template('equations_differentielles.html')

# Accueil
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
