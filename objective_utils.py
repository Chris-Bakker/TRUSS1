import pandas as pd
import math
import sys

sys.path.append('../pyJive/')
sys.path = list(set(sys.path))
import main
from utils import proputils as pu
from names import GlobNames as gn
import numpy as np


def create_element_dataframe(globdat, geom_file_path='bridge.geom', pro_file_path='bridge_frequency.pro'):
    """
    Creates a DataFrame containing information about elements, nodes, and surface areas from the .geom and .pro files.

    Parameters:
        geom_file_path (str): Path to the .geom file defining the geometry of the structure.
        pro_file_path (str): Path to the .pro file defining material and cross-section properties.
        globdat (dict): Global data structure containing node coordinates.

    Returns:
        DataFrame: A DataFrame with the following columns:
            - ID: Unique ID for each element.
            - Node1: ID of the first node in the element.
            - Coordinates_Node1: Coordinates of the first node.
            - Node2: ID of the second node in the element.
            - Coordinates_Node2: Coordinates of the second node.
            - CrossSectionType: Cross-section type of the element.
            - SurfaceArea: Surface area of the element based on the cross-section type.
    """
    areas = []
    with open(pro_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("area = ["):
                areas = eval(line.split("=")[-1].strip().strip(";"))
                break

    nodes = globdat[gn.NSET]
    elements = []
    with open(geom_file_path, 'r') as file:
        parsing_elements = False
        for line in file:
            line = line.strip()
            if line.lower().startswith("member"):
                parsing_elements = True
                continue
            if parsing_elements and line:
                parts = line.split()
                if len(parts) == 4:
                    node1, node2, num_elements, cross_section_type = map(int, parts)
                    elements.append({
                        'ID': len(elements),
                        'Node1': node1,
                        'Coordinates_Node1': nodes[node1].get_coords(),
                        'Node2': node2,
                        'Coordinates_Node2': nodes[node2].get_coords(),
                        'CrossSectionType': cross_section_type,
                        'SurfaceArea': areas[cross_section_type]
                    })

    df = pd.DataFrame(elements)
    return df


def modify_updated_geom(df, geom_updated_file_path):
    """
    Modifies an existing .geom file to update the node coordinates based on a DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing updated node coordinates and areas.
        geom_updated_file_path (str): Path to the updated .geom file.

    Returns:
        None: The modified .geom file is saved to the specified path.
    """
    with open(geom_updated_file_path, 'r') as file:
        geom_data = file.readlines()

    updated_geom_data = []
    for line in geom_data:
        if line.strip().startswith("node:") or line.strip().startswith("member:"):
            updated_geom_data.append(line)
            continue
        if line.strip():
            parts = line.strip().split()
            if len(parts) == 3:
                node_id = int(parts[0])
                updated_coords = None
                if node_id in df['Node1'].values:
                    updated_coords = df[df['Node1'] == node_id]['Coordinates_Node1'].values[0]
                elif node_id in df['Node2'].values:
                    updated_coords = df[df['Node2'] == node_id]['Coordinates_Node2'].values[0]

                if updated_coords is not None:
                    x, y = updated_coords
                    updated_geom_data.append(f"{node_id} {x} {y}\n")
                else:
                    updated_geom_data.append(line)
            else:
                updated_geom_data.append(line)

    with open(geom_updated_file_path, 'w') as file:
        file.writelines(updated_geom_data)


def update_props_with_df(props, df, geom_updated_file_path):
    """
    Updates the props dictionary with new geometry file and area array.

    Parameters:
        props (dict): The props dictionary to update.
        df (DataFrame): DataFrame containing the updated areas.
        geom_updated_file_path (str): Path to the updated .geom file.

    Returns:
        dict: The updated props dictionary.
    """
    props['init']['mesh']['file'] = geom_updated_file_path

    if isinstance(props['model']['truss']['area'], str):
        areas = eval(props['model']['truss']['area']) 
    elif isinstance(props['model']['truss']['area'], (list, np.ndarray)):
        areas = props['model']['truss']['area']
    else:
        raise TypeError(f"Unexpected type for 'area': {type(props['model']['truss']['area'])}. Must be str or list.")

    for _, row in df.iterrows():
        cross_section_type = row['CrossSectionType']
        areas[cross_section_type] = row['SurfaceArea']

    props['model']['truss']['area'] = areas
    return props


def oneD_optimization_function(y_value, df, geom_updated_file_path, props, globdat):
    """
    Performs one-dimensional optimization to calculate total weight and first three eigenfrequencies.

    Parameters:
        y_value (float): The new y-coordinate value for selected nodes.
        df (DataFrame): Initial DataFrame containing node and element information.
        geom_updated_file_path (str): Path to the updated .geom file.
        props (dict): The props dictionary to update.
        globdat (dict): The globdat structure used for calculations.

    Returns:
        tuple: Total weight and the first three eigenfrequencies.
    """
    excluded_nodes = globdat[gn.NGROUPS]['bottom']
    for index, row in df.iterrows():
        if row['Node1'] not in excluded_nodes:
            df.at[index, 'Coordinates_Node1'] = [row['Coordinates_Node1'][0], y_value]
        if row['Node2'] not in excluded_nodes:
            df.at[index, 'Coordinates_Node2'] = [row['Coordinates_Node2'][0], y_value]

    modify_updated_geom(df, geom_updated_file_path)
    props = update_props_with_df(props, df, geom_updated_file_path)
    globdat = main.jive(props)

    df['Length'] = ((df['Coordinates_Node2'].str[0] - df['Coordinates_Node1'].str[0]) ** 2 +
                    (df['Coordinates_Node2'].str[1] - df['Coordinates_Node1'].str[1]) ** 2) ** 0.5

    density = 7800
    df['Weight'] = df['SurfaceArea'] * df['Length'] * density
    total_weight = df['Weight'].sum()

    eigenfrequencies = globdat[gn.EIGENFREQS][:3] / (2 * math.pi)
    return total_weight, eigenfrequencies


def oneD_lagrange_optimization(
        y_value,
        constraint_weight,
        df,
        props,
        globdat,
        geom_updated_file_path='bridge_updated.geom',
        freq_thresholds=[20, 40, 60]
):
    """
    Performs Lagrange optimization for minimizing weight with eigenfrequency constraints.

    Parameters:
        y_value (float): The new y-coordinate value for selected nodes.
        constraint_weight (float): Weight for the constraint penalties.
        df (DataFrame): DataFrame containing node and element information.
        geom_updated_file_path (str): Path to the updated .geom file.
        props (dict): The props dictionary to update.
        globdat (dict): The globdat structure used for calculations.
        freq_thresholds (list): List of threshold eigenfrequencies [f1, f2, f3].

    Returns:
        float: The value of the Lagrange function to minimize.
    """

    weight, eigenfrequencies = oneD_optimization_function(
        y_value, df, geom_updated_file_path, props, globdat
    )
    penalty = sum(max(0, freq_thresholds[i] - eigenfrequencies[i]) ** 2 for i in range(3))
    lagrange_value = weight + constraint_weight * penalty
    return lagrange_value


def twoD_optimization_function(y_value, A_value, df, geom_updated_file_path, props, globdat):
    """
    Performs two-dimensional optimization to calculate total weight and first three eigenfrequencies.

    Parameters:
        y_value (float): The new y-coordinate value for selected nodes.
        A_value (float): The new cross-sectional area for all elements.
        df (DataFrame): Initial DataFrame containing node and element information.
        geom_updated_file_path (str): Path to the updated .geom file.
        props (dict): The props dictionary to update.
        globdat (dict): The globdat structure used for calculations.

    Returns:
        tuple: Total weight and the first three eigenfrequencies.
    """

    excluded_nodes = globdat[gn.NGROUPS]['bottom']
    for index, row in df.iterrows():
        if row['Node1'] not in excluded_nodes:
            df.at[index, 'Coordinates_Node1'] = [row['Coordinates_Node1'][0], y_value]
        if row['Node2'] not in excluded_nodes:
            df.at[index, 'Coordinates_Node2'] = [row['Coordinates_Node2'][0], y_value]

    df.loc[df['CrossSectionType'] > 0, 'SurfaceArea'] = A_value

    modify_updated_geom(df, geom_updated_file_path)
    props = update_props_with_df(props, df, geom_updated_file_path)

    globdat = main.jive(props)

    df['Length'] = ((df['Coordinates_Node2'].str[0] - df['Coordinates_Node1'].str[0]) ** 2 +
                    (df['Coordinates_Node2'].str[1] - df['Coordinates_Node1'].str[1]) ** 2) ** 0.5
    density = 7800  # Density in kg/m³
    df['Weight'] = df['SurfaceArea'] * df['Length'] * density
    total_weight = df['Weight'].sum()

    eigenfrequencies = globdat[gn.EIGENFREQS][:3] / (2 * math.pi)

    return total_weight, eigenfrequencies


def twoD_lagrange_optimization(
        y_value,
        A_value,
        constraint_weight,
        df,
        props,
        globdat,
        geom_updated_file_path='bridge_updated.geom',
        freq_thresholds=[20, 40, 60]
):
    """
    Performs Lagrange optimization for minimizing weight with eigenfrequency constraints.

    Parameters:
        y_value (float): The new y-coordinate value for selected nodes.
        A_value (float): The new cross-sectional area for all elements.
        constraint_weight (float): Weight for the constraint penalties.
        df (DataFrame): DataFrame containing node and element information.
        geom_updated_file_path (str): Path to the updated .geom file.
        props (dict): The props dictionary to update.
        globdat (dict): The globdat structure used for calculations.
        freq_thresholds (list): List of threshold eigenfrequencies [f1, f2, f3].

    Returns:
        float: The value of the Lagrange function to minimize.
    """

    A_value = A_value / 10000
    weight, eigenfrequencies = twoD_optimization_function(
        y_value, A_value, df, geom_updated_file_path, props, globdat
    )

    penalty = sum(max(0, freq_thresholds[i] - eigenfrequencies[i]) ** 2 for i in range(3))

    lagrange_value = weight + constraint_weight * penalty

    return lagrange_value


def sixD_optimization_function(y_values, A_value, df, geom_updated_file_path, props, globdat):
    """
    Performs six-dimensional optimization to calculate total weight and first three eigenfrequencies.

    Parameters:
        y1 (float): The new y-coordinate value for nodes 2 and 18.
        y2 (float): The new y-coordinate value for nodes 4 and 16.
        y3 (float): The new y-coordinate value for nodes 6 and 14.
        y4 (float): The new y-coordinate value for nodes 8 and 12.
        y5 (float): The new y-coordinate value for node 10.
        A_value (float): The new cross-sectional area for all elements.
        df (DataFrame): Initial DataFrame containing node and element information.
        geom_updated_file_path (str): Path to the updated .geom file.
        props (dict): The props dictionary to update.
        globdat (dict): The globdat structure used for calculations.

    Returns:
        tuple: Total weight and the first three eigenfrequencies.
    """
    
    A_value = A_value / 10000

    y1, y2, y3, y4, y5 = y_values
    top_nodes_mapping = {
        2: y1,
        18: y1,
        4: y2,
        16: y2,
        6: y3,
        14: y3,
        8: y4,
        12: y4,
        10: y5,
    }
    for index, row in df.iterrows():
        if row['Node1'] in top_nodes_mapping:
            df.at[index, 'Coordinates_Node1'] = [row['Coordinates_Node1'][0], top_nodes_mapping[row['Node1']]]
        if row['Node2'] in top_nodes_mapping:
            df.at[index, 'Coordinates_Node2'] = [row['Coordinates_Node2'][0], top_nodes_mapping[row['Node2']]]

    df.loc[df['CrossSectionType'] > 0, 'SurfaceArea'] = A_value

    modify_updated_geom(df, geom_updated_file_path)
    props = update_props_with_df(props, df, geom_updated_file_path)
    globdat = main.jive(props)

    df['Length'] = ((df['Coordinates_Node2'].str[0] - df['Coordinates_Node1'].str[0]) ** 2 +
                    (df['Coordinates_Node2'].str[1] - df['Coordinates_Node1'].str[1]) ** 2) ** 0.5
    density = 7800  # Density in kg/m³
    df['Weight'] = df['SurfaceArea'] * df['Length'] * density
    total_weight = df['Weight'].sum()

    eigenfrequencies = globdat[gn.EIGENFREQS][:3] / (2 * math.pi)  # Convert to Hz

    return total_weight, eigenfrequencies


def sixD_lagrange_optimization(
        y_values, A_value, constraint_weight,
        df, props, globdat, geom_updated_file_path='bridge_updated.geom',
        freq_thresholds=[20, 40, 60]
):
    """
    Performs Lagrange optimization for minimizing weight with eigenfrequency constraints in six dimensions.

    Parameters:
        y1 (float): The new y-coordinate value for nodes 2 and 18.
        y2 (float): The new y-coordinate value for nodes 4 and 16.
        y3 (float): The new y-coordinate value for nodes 6 and 14.
        y4 (float): The new y-coordinate value for nodes 8 and 12.
        y5 (float): The new y-coordinate value for node 10.
        A_value (float): The new cross-sectional area for all elements.
        constraint_weight (float): Weight for the constraint penalties.
        df (DataFrame): DataFrame containing node and element information.
        geom_updated_file_path (str): Path to the updated .geom file.
        props (dict): The props dictionary to update.
        globdat (dict): The globdat structure used for calculations.
        freq_thresholds (list): List of threshold eigenfrequencies [f1, f2, f3].

    Returns:
        float: The value of the Lagrange function to minimize.
    """
    weight, eigenfrequencies = sixD_optimization_function(
        y_values, A_value, df, geom_updated_file_path, props, globdat
    )

    penalty = sum(max(0, freq_thresholds[i] - eigenfrequencies[i]) ** 2 for i in range(3))

    lagrange_value = weight + constraint_weight * penalty

    return lagrange_value


def nineteenD_optimization_function(
        y_values, A_values, df, geom_updated_file_path, props, globdat
):
    """
    Performs 19D optimization to calculate total weight and first three eigenfrequencies.

    Parameters:
        y_values (list): List of 5 y-coordinate values for nodes [2, 18], [4, 16], [6, 14], [8, 12], and [10].
        A_values (list): List of 14 surface area values for each CrossSectionType (1–14).
        df (DataFrame): Initial DataFrame containing node and element information.
        geom_updated_file_path (str): Path to the updated .geom file.
        props (dict): The props dictionary to update.
        globdat (dict): The globdat structure used for calculations.

    Returns:
        tuple: Total weight and the first three eigenfrequencies.
    """
    y1, y2, y3, y4, y5 = y_values
    top_nodes_mapping = {
        2: y1,
        18: y1,
        4: y2,
        16: y2,
        6: y3,
        14: y3,
        8: y4,
        12: y4,
        10: y5,
    }
    for index, row in df.iterrows():
        if row['Node1'] in top_nodes_mapping:
            df.at[index, 'Coordinates_Node1'] = [row['Coordinates_Node1'][0], top_nodes_mapping[row['Node1']]]
        if row['Node2'] in top_nodes_mapping:
            df.at[index, 'Coordinates_Node2'] = [row['Coordinates_Node2'][0], top_nodes_mapping[row['Node2']]]

    A_values = [A / 10000 for A in A_values]
    for i in range(len(A_values)):
        df.loc[df['CrossSectionType'] == i + 1, 'SurfaceArea'] = A_values[i]

    modify_updated_geom(df, geom_updated_file_path)
    props = update_props_with_df(props, df, geom_updated_file_path)
    globdat = main.jive(props)

    df['Length'] = ((df['Coordinates_Node2'].str[0] - df['Coordinates_Node1'].str[0]) ** 2 +
                    (df['Coordinates_Node2'].str[1] - df['Coordinates_Node1'].str[1]) ** 2) ** 0.5
    density = 7800
    df['Weight'] = df['SurfaceArea'] * df['Length'] * density
    total_weight = df['Weight'].sum()

    eigenfrequencies = globdat[gn.EIGENFREQS][:3] / (2 * math.pi)  # Convert to Hz

    return total_weight, eigenfrequencies


def nineteenD_lagrange_optimization(
        y_values, A_values, constraint_weight, df, props, globdat,
        geom_updated_file_path='bridge_updated.geom', freq_thresholds=[20, 40, 60]
):
    """
    Performs Lagrange optimization for minimizing weight with eigenfrequency constraints in 19 dimensions.

    Parameters:
        y_values (list): List of 5 y-coordinate values for nodes [2, 18], [4, 16], [6, 14], [8, 12], and [10].
        A_values (list): List of 14 surface area values for each CrossSectionType (1–14).
        constraint_weight (float): Weight for the constraint penalties.
        df (DataFrame): DataFrame containing node and element information.
        geom_updated_file_path (str): Path to the updated .geom file.
        props (dict): The props dictionary to update.
        globdat (dict): The globdat structure used for calculations.
        freq_thresholds (list): List of threshold eigenfrequencies [f1, f2, f3].

    Returns:
        float: The value of the Lagrange function to minimize.
    """
    weight, eigenfrequencies = nineteenD_optimization_function(
        y_values, A_values, df, geom_updated_file_path, props, globdat
    )

    penalty = sum(max(0, freq_thresholds[i] - eigenfrequencies[i]) ** 2 for i in range(3))

    lagrange_value = weight + constraint_weight * penalty

    return lagrange_value




