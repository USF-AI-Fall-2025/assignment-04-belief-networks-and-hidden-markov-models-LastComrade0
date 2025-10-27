from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

car_model = DiscreteBayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("KeyPresent", "Starts"),
        ("Starts","Moves"),
])

# Defining the parameters using CPT


cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_keypresent = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]],
    state_names={"KeyPresent":['yes',"no"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

# Update CPD with KeyPresent
cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], 
            [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
    evidence=["Gas", "Ignition", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={"Starts":['yes','no'], 
                 "Gas":['Full',"Empty"], 
                 "Ignition":["Works", "Doesn't work"],
                 "KeyPresent":['yes','no']},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_keypresent)

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))


def main():
    print("=== Car Network Queries ===")

    # Query 1: Battery not working given car will not move
    q1 = car_infer.query(variables=["Battery"], evidence={"Moves":"no"})
    print("\n1. P(Battery doesn't work | Car won't move): \n", q1)

    # Query 2: Car will not start given radio not working
    q2 = car_infer.query(variables=["Starts"], evidence={"Radio":"Doesn't turn on"})
    print("\n2. P(Car won't start | Radio doesn't turn on): \n", q2)

    # Query 3: Probability of the radio working change if we discover that the car has gas in it? Given battery is working
    q3a = car_infer.query(variables=["Radio"], evidence={"Battery":"Works"})
    q3b = car_infer.query(variables=["Radio"], evidence={"Battery":"Works", "Gas":"Full"})
    print("\n3a. P(Radio | Battery=works): \n", q3a)
    print("\n3b. P(Radio | Battery=works, Gas=full): \n", q3b)

    # Query 4: Ignition failing given car doesn't move, with and without gas
    q4a = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    q4b = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
    print("\n4a. P(Ignition | Moves=no): \n", q4a)
    print("\n4b. P(Ignition | Moves=no, Gas=Empty): \n", q4b)
    print("\nHow does the probability change? Compare the values above.\n")

    # Query 5: Car starts given radio works and has gas
    q5 = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
    print("\n5. P(Car starts | Radio=turns on, Gas=full): \n", q5)

    q6 = car_infer.query(variables=["KeyPresent"], evidence={"Moves": "no"})
    print("\n6. P(KeyPresent | Moves=no): \n", q6)


if __name__ == "__main__":
    main()