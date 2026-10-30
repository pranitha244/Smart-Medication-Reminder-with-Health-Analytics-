from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from pathlib import Path
import sqlite3, pickle, numpy as np, os

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'local-development-only-change-me')
DB = BASE / 'patients.db'
with open(BASE / 'smr_model.pkl', 'rb') as f:
    saved = pickle.load(f)  # Load only the trusted model bundled with this project
model, scaler = saved['model'], saved['scaler']
ge, me, ce = saved['gender_encoder'], saved['medication_encoder'], saved['condition_encoder']

def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with db() as con:
        con.executescript("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS patients(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, name TEXT NOT NULL, age INTEGER NOT NULL, gender TEXT NOT NULL, condition TEXT NOT NULL, medication TEXT NOT NULL, dosage REAL NOT NULL);
CREATE TABLE IF NOT EXISTS schedules(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, patient_id INTEGER NOT NULL, time TEXT NOT NULL, notes TEXT DEFAULT '', active INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, patient_id INTEGER NOT NULL, probability REAL NOT NULL, risk TEXT NOT NULL, created_at TEXT NOT NULL);""")
init_db()

def auth():
    return 'uid' in session

def rows(query, params=()):
    with db() as con:
        return [dict(x) for x in con.execute(query, params).fetchall()]

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name=request.form.get('name','').strip(); email=request.form.get('email','').strip().lower(); password=request.form.get('password','')
        if not name or '@' not in email or len(password)<8:
            flash('Enter a name, valid email and password of at least 8 characters.'); return redirect(url_for('register'))
        try:
            with db() as con:
                con.execute('INSERT INTO users(name,email,password) VALUES(?,?,?)',(name,email,generate_password_hash(password)))
            flash('Account created. Please sign in.'); return redirect(url_for('login'))
        except sqlite3.IntegrityError: flash('Email already registered.')
    return render_template('auth.html', mode='register')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        user=rows('SELECT * FROM users WHERE email=?',(request.form.get('email','').lower().strip(),))
        if user and check_password_hash(user[0]['password'],request.form.get('password','')):
            session.clear();session['uid']=user[0]['id'];session['name']=user[0]['name'];return redirect(url_for('home'))
        flash('Invalid email or password.')
    return render_template('auth.html',mode='login')

@app.route('/logout')
def logout():
    session.clear();return redirect(url_for('login'))

@app.route('/')
def home():
    if not auth(): return redirect(url_for('login'))
    uid=session['uid']; patients=rows('SELECT * FROM patients WHERE user_id=? ORDER BY id DESC',(uid,))
    schedules=rows('SELECT s.*,p.name AS patient_name,p.medication FROM schedules s JOIN patients p ON s.patient_id=p.id WHERE s.user_id=? ORDER BY s.time',(uid,))
    predictions=rows('SELECT r.*,p.name AS patient_name FROM predictions r JOIN patients p ON p.id=r.patient_id WHERE r.user_id=? ORDER BY r.id DESC LIMIT 30',(uid,))
    counts={k:sum(p['risk']==k for p in predictions) for k in ('High','Moderate','Low')}
    return render_template('dashboard.html',patients=patients,schedules=schedules,predictions=predictions,counts=counts,medications=list(me.classes_),conditions=list(ce.classes_))

@app.route('/patient',methods=['POST'])
def patient():
    if not auth(): return redirect(url_for('login'))
    try:
        age=int(request.form['age']); dosage=float(request.form['dosage']); gender=request.form['gender']; med=request.form['medication']; condition=request.form['condition']; name=request.form['name'].strip()
        if not name or not 1<=age<=120 or not 0<dosage<=100000 or gender not in ('Male','Female') or med not in me.classes_ or condition not in ce.classes_: raise ValueError()
        with db() as con: con.execute('INSERT INTO patients(user_id,name,age,gender,condition,medication,dosage) VALUES(?,?,?,?,?,?,?)',(session['uid'],name,age,gender,condition,med,dosage))
        flash('Patient added.')
    except (ValueError,KeyError): flash('Please check the patient details.')
    return redirect(url_for('home'))

@app.route('/schedule',methods=['POST'])
def schedule():
    if not auth(): return redirect(url_for('login'))
    try:
        pid=int(request.form['patient_id']); time=request.form['time']; datetime.strptime(time,'%H:%M')
        if not rows('SELECT id FROM patients WHERE id=? AND user_id=?',(pid,session['uid'])): raise ValueError()
        with db() as con: con.execute('INSERT INTO schedules(user_id,patient_id,time,notes) VALUES(?,?,?,?)',(session['uid'],pid,time,request.form.get('notes','')[:200]))
        flash('Reminder scheduled. Keep this page open to see browser reminders.')
    except (ValueError,KeyError): flash('Invalid reminder details.')
    return redirect(url_for('home'))

@app.route('/predict/<int:pid>',methods=['POST'])
def predict(pid):
    if not auth(): return redirect(url_for('login'))
    p=rows('SELECT * FROM patients WHERE id=? AND user_id=?',(pid,session['uid']))
    if not p: flash('Patient not found.');return redirect(url_for('home'))
    p=p[0]
    try:
        reminder=int(request.form.get('reminder_sent','1'))
        if reminder not in (0,1): raise ValueError()
        gender='M' if p['gender']=='Male' else 'F'
        features=np.array([[p['age'],ge.transform([gender])[0],me.transform([p['medication']])[0],ce.transform([p['condition']])[0],p['dosage'],reminder]])
        scaled=scaler.transform(features)
        classes=list(model.classes_)
        probability=float(model.predict_proba(scaled)[0][classes.index(1)])
        risk='High' if probability>0.55 else 'Moderate' if probability>0.30 else 'Low'
        with db() as con: con.execute('INSERT INTO predictions(user_id,patient_id,probability,risk,created_at) VALUES(?,?,?,?,?)',(session['uid'],pid,probability,risk,datetime.now().strftime('%Y-%m-%d %H:%M')))
        flash(f"{p['name']}: {risk} estimated non-adherence risk ({probability:.1%}). Educational estimate only.")
    except Exception as e:
        app.logger.exception('Prediction error');flash('Prediction failed. Check model compatibility and input data.')
    return redirect(url_for('home'))

if __name__=='__main__':
    app.run(debug=False)
