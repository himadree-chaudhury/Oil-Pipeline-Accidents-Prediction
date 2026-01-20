import gradio as gr
import pandas as pd
import pickle

print("Loading model...")
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

# *Category Options
pipeline_types = [
    'ABOVEGROUND', 'TANK', 'TRANSITION AREA', 'UNDERGROUND'
]

liquid_types = [
    'BIOFUEL / ALTERNATIVE FUEL(INCLUDING ETHANOL BLENDS)', 
    'CO2 (CARBON DIOXIDE)', 
    'CRUDE OIL', 
    'HVL OR OTHER FLAMMABLE OR TOXIC FLUID, GAS', 
    'REFINED AND/OR PETROLEUM PRODUCT (NON-HVL), LIQUID'
]

states = [
    'AK', 'AL', 'AR', 'CA', 'CO', 'CT', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 
    'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 
    'NE', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'SC', 'SD', 'TN', 
    'TX', 'UT', 'VA', 'WA', 'WI', 'WV', 'WY'
]

cause_categories = [
    'ALL OTHER CAUSES', 'CORROSION', 'EXCAVATION DAMAGE', 'INCORRECT OPERATION', 
    'MATERIAL/WELD/EQUIP FAILURE', 'NATURAL FORCE DAMAGE', 'OTHER OUTSIDE FORCE DAMAGE'
]

cause_subcategories = [
    'CONSTRUCTION, INSTALLATION OR FABRICATION-RELATED', "DAMAGE BY OPERATOR OR OPERATOR'S CONTRACTOR", 
    'DEFECTIVE OR LOOSE TUBING/FITTING', 'EARTH MOVEMENT', 'ELECTRICAL ARCING FROM OTHER EQUIPMENT/FACILITY', 
    'ENVIRONMENTAL CRACKING-RELATED', 'EXTERNAL', 'FAILURE OF EQUIPMENT BODY', 
    'FIRE/EXPLOSION AS PRIMARY CAUSE', 'FISHING OR MARITIME ACTIVITY', 'HEAVY RAINS/FLOODS', 
    'HIGH WINDS', 'INCORRECT EQUIPMENT', 'INCORRECT INSTALLATION', 'INCORRECT VALVE POSITION', 
    'INTENTIONAL DAMAGE', 'INTERNAL', 'LIGHTNING', 'MALFUNCTION OF CONTROL/RELIEF EQUIPMENT', 
    'MANUFACTURING-RELATED', 'MARITIME EQUIPMENT OR VESSEL ADRIFT', 'MISCELLANEOUS', 
    'NON-THREADED CONNECTION FAILURE', 'OPERATOR/CONTRACTOR EXCAVATION DAMAGE', 
    'OTHER EQUIPMENT FAILURE', 'OTHER INCORRECT OPERATION', 'OTHER NATURAL FORCE DAMAGE', 
    'OTHER OUTSIDE FORCE DAMAGE', 'OVERFILL/OVERFLOW OF TANK/VESSEL/SUMP', 
    'PIPELINE/EQUIPMENT OVERPRESSURED', 'PREVIOUS DAMAGE DUE TO EXCAVATION', 
    'PREVIOUS MECHANICAL DAMAGE', 'PUMP OR PUMP-RELATED EQUIPMENT', 'TEMPERATURE', 
    'THIRD PARTY EXCAVATION DAMAGE', 'THREADED CONNECTION/COUPLING FAILURE', 'UNKNOWN', 
    'VEHICLE NOT ENGAGED IN EXCAVATION'
]

shutdown_options = ['NO', 'YES']

# *Prediction Function
def predict_release(pipeline_type, liquid_type, state, latitude, longitude, 
                    cause_category, cause_subcategory, recovery_barrels, shutdown):
    
    input_data = pd.DataFrame({
        'Pipeline Type': [pipeline_type],
        'Liquid Type': [liquid_type],
        'Accident State': [state],
        'Accident Latitude': [latitude],
        'Accident Longitude': [longitude],
        'Cause Category': [cause_category],
        'Cause Subcategory': [cause_subcategory],
        'Liquid Recovery (Barrels)': [recovery_barrels],
        'Pipeline Shutdown': [shutdown]
    })
    
    prediction = model.predict(input_data)
    
    result = max(0, prediction[0])
    return f"{result:.2f} Barrels"

# *Gradio Interface
inputs = [
    gr.Dropdown(label="Pipeline Type", choices=pipeline_types, value=pipeline_types[0]),
    gr.Dropdown(label="Liquid Type", choices=liquid_types, value=liquid_types[2]),
    gr.Dropdown(label="Accident State", choices=states, value="TX"),
    gr.Number(label="Accident Latitude", value=29.76),
    gr.Number(label="Accident Longitude", value=-95.36),
    gr.Dropdown(label="Cause Category", choices=cause_categories, value=cause_categories[1]),
    gr.Dropdown(label="Cause Subcategory", choices=cause_subcategories, value=cause_subcategories[16]),
    gr.Number(label="Liquid Recovery (Barrels)", value=0),
    gr.Radio(label="Pipeline Shutdown", choices=shutdown_options, value="YES")
]

outputs = gr.Textbox(label="Predicted Unintentional Release")

app = gr.Interface(
    fn=predict_release,
    inputs=inputs,
    outputs=outputs,
    title="Pipeline Accident Release Predictor",
    description="Predict the volume of unintentional release (in Barrels) based on accident details using a Linear Regression model.",
    examples=[
        ['ABOVEGROUND', 'CRUDE OIL', 'TX', 30.0, -95.0, 'CORROSION', 'INTERNAL', 10, 'YES'],
        ['UNDERGROUND', 'HVL OR OTHER FLAMMABLE OR TOXIC FLUID, GAS', 'LA', 30.5, -91.0, 'MATERIAL/WELD/EQUIP FAILURE', 'PUMP OR PUMP-RELATED EQUIPMENT', 0, 'NO']
    ]
)

app.launch(share=True)