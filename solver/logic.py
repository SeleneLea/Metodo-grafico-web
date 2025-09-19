import numpy as np
from itertools import combinations

def solve_graphical_method(data):
    # --- 1.  ---
    # extraer la función objetivo y el objetivo (maximizar/minimizar)
    goal = data.get('goal', 'maximize')
    obj_coeffs = [float(data.get('obj_x1', 0)), float(data.get('obj_x2', 0))]

    # extraer las restricciones del formulario
    constraints = []
    i = 1
    while True: # Un bucle infinito que romperemos nosotros
        try:
            # Intenta acceder a los datos de la restricción 'i'
            x1_coeff = data[f'c{i}_x1']
            
            # Si el campo está vacío, asumimos que no hay más restricciones
            if not x1_coeff:
                break

            constraint = {
                'coeffs': [float(x1_coeff), float(data[f'c{i}_x2'])],
                'op': data[f'c{i}_op'],
                'rhs': float(data[f'c{i}_rhs'])
            }
            if constraint['coeffs'][0] != 0 or constraint['coeffs'][1] != 0:
                constraints.append(constraint)
            
            i += 1 # Pasamos a la siguiente restricción
        except (KeyError, ValueError):
            # Si no se encuentra una clave (ej: 'c5_x1'), significa que no hay más.
            break

    # añadir restricciones de no-negatividad (x1 >= 0, x2 >= 0)
    constraints.append({'coeffs': [1, 0], 'op': '>=', 'rhs': 0}) # x1 >= 0
    constraints.append({'coeffs': [0, 1], 'op': '>=', 'rhs': 0}) # x2 >= 0

    # --- 2. FIND INTERSECTION POINTS ---
    intersection_points = []
    for c1, c2 in combinations(constraints, 2):
        A = np.array([c1['coeffs'], c2['coeffs']])
        B = np.array([c1['rhs'], c2['rhs']])
        
        if np.linalg.det(A) != 0:
            try:
                solution = np.linalg.solve(A, B)
                if solution[0] >= -1e-6 and solution[1] >= -1e-6:
                    intersection_points.append(tuple(solution))
            except np.linalg.LinAlgError:
                continue
    
    # --- 3. IDENTIFY FEASIBLE VERTICES ---
    feasible_vertices = []
    for point in set(intersection_points):
        x1, x2 = point
        is_feasible = True
        for c in constraints:
            val = c['coeffs'][0] * x1 + c['coeffs'][1] * x2
            op = c['op']
            rhs = c['rhs']
            
            if (op == '<=' and val > rhs + 1e-6) or \
               (op == '>=' and val < rhs - 1e-6):
                is_feasible = False
                break
        if is_feasible:
            feasible_vertices.append(point)

    if not feasible_vertices:
        return {'error': 'No se encontró una región factible.'}

    # --- 4. FIND OPTIMAL SOLUTION ---
    best_value = None
    optimal_point = None

    for vertex in feasible_vertices:
        value = obj_coeffs[0] * vertex[0] + obj_coeffs[1] * vertex[1]
        if best_value is None:
            best_value = value
            optimal_point = vertex
        elif goal == 'maximize' and value > best_value:
            best_value = value
            optimal_point = vertex
        elif goal == 'minimize' and value < best_value:
            best_value = value
            optimal_point = vertex

    # --- 5. PREPARE DATA FOR GRAPHING ---
    if not optimal_point:
         return {'error': 'No se encontró un punto óptimo en la región factible.'}
         
    center = np.mean(feasible_vertices, axis=0)
    sorted_vertices = sorted(feasible_vertices, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    
    plot_lines = []
    max_val = max(p[0] for p in feasible_vertices + [(0,0)]) * 1.2
    if max_val == 0: max_val = 10
    
    for c in constraints:
      if c['coeffs'] == [1,0] or c['coeffs'] == [0,1]:
        continue
      
      points = []
      if c['coeffs'][1] != 0:
          y_intercept = c['rhs'] / c['coeffs'][1]
          points.append({'x': 0, 'y': y_intercept})

      if c['coeffs'][0] != 0:
          x_intercept = c['rhs'] / c['coeffs'][0]
          points.append({'x': x_intercept, 'y': 0})
      
      if len(points) < 2 and c['coeffs'][0] != 0 and c['coeffs'][1] != 0:
        y_val_at_max_x = (c['rhs'] - c['coeffs'][0] * max_val) / c['coeffs'][1]
        points.append({'x': max_val, 'y': y_val_at_max_x})

      plot_lines.append(points)

    return {
        'feasible_vertices': [{'x': v[0], 'y': v[1]} for v in sorted_vertices],
        'optimal_point': {'x': optimal_point[0], 'y': optimal_point[1]},
        'optimal_value': round(best_value, 2),
        'plot_lines': plot_lines
    }