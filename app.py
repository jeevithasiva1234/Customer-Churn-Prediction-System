import os
import random
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
app.secret_key = 'demo-secret-key-2026'

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'customers.csv')

GENDER_MAP = {'Male': 0, 'Female': 1, 'Other': 2}
REGION_MAP = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Central': 4}
ACTIVE_MAP = {'Yes': 1, 'No': 0}

USERS = {
    'admin': {'password': 'admin123', 'role': 'Admin'},
    'staff': {'password': 'staff123', 'role': 'Staff'},
}


def create_sample_data():
    # Generate a simple 110-row sample dataset for the demo
    names = [
        'Ariana', 'Ben', 'Chloe', 'Dylan', 'Eva', 'Felix', 'Grace', 'Hannah', 'Ivan', 'Jade',
        'Kiran', 'Lina', 'Maya', 'Nina', 'Owen', 'Priya', 'Quinn', 'Ria', 'Sanjay', 'Tara',
        'Uma', 'Vik', 'Will', 'Xena', 'Yara', 'Zane'
    ]
    genders = ['Male', 'Female', 'Other']
    regions = ['North', 'South', 'East', 'West', 'Central']
    rows = []
    for i in range(1, 111):
        name = random.choice(names) + ' ' + random.choice(['Lee', 'Shah', 'Patel', 'Miller', 'Roy', 'Kim'])
        age = random.randint(21, 63)
        gender = random.choice(genders)
        region = random.choice(regions)
        monthly_charges = round(random.uniform(20.0, 120.0), 2)
        tenure_months = random.randint(1, 48)
        support_tickets = random.randint(0, 8)
        is_active = random.choice(['Yes'] * 7 + ['No'] * 3)
        churn = 1 if (is_active == 'No' or monthly_charges > 90 or tenure_months < 8 or support_tickets > 4) else 0
        rows.append({
            'customer_id': f'C{i:03d}',
            'name': name,
            'age': age,
            'gender': gender,
            'region': region,
            'monthly_charges': monthly_charges,
            'tenure_months': tenure_months,
            'support_tickets': support_tickets,
            'is_active': is_active,
            'churn': churn,
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)


def load_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        create_sample_data()
    return pd.read_csv(DATA_PATH)


def save_data(df):
    df.to_csv(DATA_PATH, index=False)


def encode_input(value, mapping):
    return mapping.get(value, 0)


def login_required(route):
    @wraps(route)
    def wrapped(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return route(*args, **kwargs)
    return wrapped


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        user = USERS.get(username)
        if user and user['password'] == password:
            session['user'] = username.capitalize()
            session['role'] = user['role']
            return redirect(url_for('dashboard'))
        error = 'Invalid username or password'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


@app.context_processor
def inject_user():
    return dict(logged_in=session.get('user'), user_role=session.get('role'))


def get_model():
    df = load_data()
    features = ['age', 'gender', 'region', 'monthly_charges', 'tenure_months', 'support_tickets', 'is_active']
    df_encoded = df.copy()
    df_encoded['gender'] = df_encoded['gender'].map(GENDER_MAP)
    df_encoded['region'] = df_encoded['region'].map(REGION_MAP)
    df_encoded['is_active'] = df_encoded['is_active'].map(ACTIVE_MAP)
    X = df_encoded[features]
    y = df_encoded['churn']
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model


model = get_model()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/customers')
@login_required
def customers():
    search = request.args.get('search', '').strip()
    gender_filter = request.args.get('gender', '')
    region_filter = request.args.get('region', '')
    df = load_data()
    if search:
        df = df[df['name'].str.contains(search, case=False, na=False)]
    if gender_filter:
        df = df[df['gender'] == gender_filter]
    if region_filter:
        df = df[df['region'] == region_filter]

    genders = sorted(load_data()['gender'].unique())
    regions = sorted(load_data()['region'].unique())
    return render_template('customers.html', customers=df.to_dict(orient='records'), genders=genders, regions=regions, search=search, selected_gender=gender_filter, selected_region=region_filter)


@app.route('/customers/add', methods=['POST'])
@login_required
def add_customer():
    df = load_data()
    next_id = f"C{len(df) + 1:03d}"
    new_customer = {
        'customer_id': next_id,
        'name': request.form.get('name', 'Unknown').strip().title(),
        'age': int(request.form.get('age', 0)),
        'gender': request.form.get('gender', 'Other'),
        'region': request.form.get('region', 'Central'),
        'monthly_charges': float(request.form.get('monthly_charges', 0.0)),
        'tenure_months': int(request.form.get('tenure_months', 0)),
        'support_tickets': int(request.form.get('support_tickets', 0)),
        'is_active': request.form.get('is_active', 'Yes'),
        'churn': 1 if request.form.get('is_active', 'Yes') == 'No' else 0,
    }
    df = pd.concat([df, pd.DataFrame([new_customer])], ignore_index=True)
    save_data(df)
    return redirect(url_for('customers'))


@app.route('/customers/delete/<customer_id>')
@login_required
def delete_customer(customer_id):
    df = load_data()
    df = df[df['customer_id'] != customer_id]
    save_data(df)
    return redirect(url_for('customers'))


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    prediction = None
    risk = None
    probability = None
    message = None
    form_data = {
        'name': '',
        'age': '',
        'gender': 'Female',
        'region': 'North',
        'tenure_months': '',
        'monthly_charges': '',
        'support_tickets': '',
        'is_active': 'Yes',
    }

    if request.method == 'POST':
        form_data.update(request.form)
        try:
            features = [
                int(form_data['age']),
                encode_input(form_data['gender'], GENDER_MAP),
                encode_input(form_data['region'], REGION_MAP),
                float(form_data['monthly_charges']),
                int(form_data['tenure_months']),
                int(form_data['support_tickets']),
                encode_input(form_data['is_active'], ACTIVE_MAP),
            ]
            prob = model.predict_proba([features])[0][1]
            probability = round(prob * 100, 1)
            prediction = 'Churn' if prob >= 0.5 else 'Stay'
            if probability >= 65:
                risk = 'High'
                message = 'The customer is at high risk. Review their plan and improve support quickly.'
            elif probability >= 40:
                risk = 'Medium'
                message = 'The customer has a moderate risk. Keep communication active and offer incentives.'
            else:
                risk = 'Low'
                message = 'The customer is likely to stay. Continue maintaining good service.'
        except Exception:
            prediction = 'Invalid Input'
            risk = 'Unknown'
            probability = 0
            message = 'Please fill all fields carefully to obtain a prediction.'

    return render_template('predict.html', form=form_data, prediction=prediction, risk=risk, probability=probability, message=message)


@app.route('/dashboard')
@login_required
def dashboard():
    df = load_data()
    total_customers = len(df)
    active_customers = len(df[df['is_active'] == 'Yes'])
    churned_customers = len(df[df['churn'] == 1])
    churn_rate = round((churned_customers / total_customers) * 100, 1) if total_customers else 0

    churn_counts = df['churn'].value_counts().sort_index().reindex([0, 1], fill_value=0).tolist()
    churn_labels = ['Stay', 'Churn']

    gender_groups = df.groupby('gender')['churn'].sum().reindex(['Male', 'Female', 'Other'], fill_value=0)
    gender_labels = gender_groups.index.tolist()
    gender_values = gender_groups.tolist()

    region_counts = df['region'].value_counts().sort_values(ascending=False)
    region_labels = region_counts.index.tolist()
    region_values = region_counts.tolist()

    bins = [0, 30, 60, 90, 120]
    monthly_labels = ['0-30', '31-60', '61-90', '91-120']
    df['charge_bin'] = pd.cut(df['monthly_charges'], bins=bins, labels=monthly_labels, include_lowest=True)
    monthly_counts = df['charge_bin'].value_counts().reindex(monthly_labels, fill_value=0).tolist()

    tenure_groups = pd.cut(df['tenure_months'], bins=[0, 12, 24, 36, 48], labels=['0-12', '13-24', '25-36', '37-48'])
    tenure_churn = df.groupby(tenure_groups)['churn'].mean().fillna(0).round(2).tolist()
    tenure_labels = ['0-12', '13-24', '25-36', '37-48']

    return render_template(
        'dashboard.html',
        total_customers=total_customers,
        active_customers=active_customers,
        churned_customers=churned_customers,
        churn_rate=churn_rate,
        churn_labels=churn_labels,
        churn_counts=churn_counts,
        gender_labels=gender_labels,
        gender_values=gender_values,
        region_labels=region_labels,
        region_values=region_values,
        monthly_labels=monthly_labels,
        monthly_counts=monthly_counts,
        tenure_labels=tenure_labels,
        tenure_churn=tenure_churn,
    )


if __name__ == '__main__':
    app.run(debug=True)
