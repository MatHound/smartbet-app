import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from scipy.stats import poisson
from datetime import datetime, timedelta

# ==============================================================================
# 1. CONFIGURAZIONE
# ==============================================================================
st.set_page_config(page_title="SmartBet Pro 60.0 Final Fix", page_icon="⚡", layout="wide")

STAGIONE = "2526"
REGION = 'eu'
MARKET = 'h2h'
HISTORY_FILE = "smartbet_odds_history.csv" 

# CSS Custom (Terminal Style)
st.markdown("""
<style>
    .terminal-box { font-family: "Courier New", Courier, monospace; background-color: #0c0c0c; color: #cccccc; padding: 15px; border-radius: 5px; border: 1px solid #333; white-space: pre; overflow-x: auto; font-size: 0.9em; margin-bottom: 10px; }
    .term-header { color: #FFD700; font-weight: bold; } 
    .term-section { color: #00FFFF; font-weight: bold; margin-top: 10px; display: block; } 
    .term-green { color: #00FF00; font-weight: bold; } 
    .term-val { color: #FF00FF; font-weight: bold; } 
    .term-warn { color: #FF4500; font-weight: bold; background-color: #330000; padding: 2px; }
    .term-fatigue { color: #FFA500; font-weight: bold; border: 1px solid #FFA500; padding: 2px; }
    .term-drop { color: #00FF00; font-weight: bold; border: 1px solid #00FF00; padding: 2px; }
    .term-dim { color: #555555; }
</style>
""", unsafe_allow_html=True)

# Session State
if 'results_data' not in st.session_state: st.session_state['results_data'] = {}

# --- DATABASE LEGHE ---
LEAGUE_GROUPS = {
    "🏆 Top 5 (Tier 1)": ['I1', 'E0', 'SP1', 'D1', 'F1'],
    "⚽ Europe Tier 2": ['N1', 'P1', 'B1', 'T1', 'SC0', 'G1', 'A1', 'SW1'],
    "📉 Leghe Minori": ['I2', 'E1', 'E2', 'E3', 'EC', 'D2', 'SP2', 'F2', 'SC1', 'SC2', 'SC3'],
    "🌎 Extra": ['ARG', 'BRA', 'CHN', 'DNK', 'FIN', 'IRL', 'JPN', 'MEX', 'NOR', 'POL', 'ROU', 'RUS', 'SWE', 'USA']
}

ALL_LEAGUES = {
    'I1': 'Serie A', 'E0': 'Premier League', 'SP1': 'La Liga', 'D1': 'Bundesliga', 'F1': 'Ligue 1',
    'N1': 'Eredivisie', 'P1': 'Primeira Liga', 'B1': 'Pro League', 'T1': 'Super Lig',
    'SC0': 'Premiership', 'G1': 'Super League', 'A1': 'Bundesliga AT', 'SW1': 'Super League CH'
}

API_MAPPING = {
    'I1': 'soccer_italy_serie_a', 'E0': 'soccer_epl', 'SP1': 'soccer_spain_la_liga', 
    'D1': 'soccer_germany_bundesliga', 'F1': 'soccer_france_ligue_one'
}

LEAGUE_COEFF = {'E0': 1.0, 'SP1': 0.96, 'I1': 0.94, 'D1': 0.92, 'F1': 0.88}

TEAM_MAPPING = {
    'Austria Wien': 'Austria Vienna', 'FC Blau-Weiß Linz': 'BW Linz', 'Grazer AK': 'GAK',
    'Hartberg': 'Hartberg', 'LASK': 'LASK Linz', 'RB Salzburg': 'Salzburg', 'Red Bull Salzburg': 'Salzburg',
    'Rapid Wien': 'Rapid Vienna', 'Rheindorf Altach': 'Altach', 'Ried': 'Ried',
    'Sturm Graz': 'Sturm Graz', 'SK Sturm Graz': 'Sturm Graz', 'WSG Tirol': 'Tirol', 'Wolfsberger AC': 'Wolfsberger',
    'Young Boys': 'Young Boys', 'Basel': 'Basel', 'FC Lausanne-Sport': 'Lausanne',
    'Lugano': 'Lugano', 'Luzern': 'Luzern', 'Sion': 'Sion', 'FC St Gallen': 'St Gallen', 'Thun': 'Thun',
    'Winterthur': 'Winterthur', 'Zurich': 'Zurich', 'Grasshopper': 'Grasshoppers', 'Servette': 'Servette',
    'Atlético Madrid': 'Ath Madrid', 'Espanyol': 'Espanol', 'RCD Espanyol': 'Espanol', 'Real Sociedad B': 'Sociedad B',
    'Südtirol': 'Sudtirol', 'US Catanzaro 1929': 'Catanzaro', 'SC Preußen Münster': 'Preussen Munster', 'VfL Bochum': 'Bochum', 
    '1. FC Heidenheim': 'Heidenheim', 'Holstein Kiel': 'Holstein Kiel', 'FC St. Pauli': 'St Pauli',
    'AE Kifisia FC': 'Kifisia', 'Levadiakos': 'Levadeiakos', 'Panetolikos Agrinio': 'Panetolikos', 'Volos FC': 'Volos NFC',
    'AVS Futebol SAD': 'AVS', 'Wimbledon': 'AFC Wimbledon', 'Basaksehir': 'Buyuksehyr', 'Istanbul Basaksehir': 'Buyuksehyr', 'Goztepe': 'Goztep', 'Besiktas JK': 'Besiktas',
    'Inter Milan': 'Inter', 'AC Milan': 'Milan', 'Napoli': 'Napoli', 'Juventus': 'Juventus', 'Atalanta BC': 'Atalanta', 'Hellas Verona': 'Verona', 'Udinese Calcio': 'Udinese', 
    'Cagliari Calcio': 'Cagliari', 'US Lecce': 'Lecce', 'Empoli FC': 'Empoli', 'Sassuolo Calcio': 'Sassuolo', 'Salernitana': 'Salernitana', 'Monza': 'Monza', 
    'Frosinone': 'Frosinone', 'Genoa': 'Genoa', 'Parma': 'Parma', 'Como': 'Como', 'Venezia': 'Venezia', 'Pisa': 'Pisa', 'Cremonese': 'Cremonese', 'Palermo': 'Palermo', 
    'Bari': 'Bari', 'Sampdoria': 'Sampdoria', 'Spezia Calcio': 'Spezia', 'Modena FC': 'Modena', 'Catanzaro': 'Catanzaro', 'Reggiana': 'Reggiana', 'Brescia': 'Brescia',
    'Cosenza': 'Cosenza', 'Sudtirol': 'Sudtirol', 'Cittadella': 'Cittadella', 'Mantova': 'Mantova', 'Cesena FC': 'Cesena', 'Cesena': 'Cesena', 'Juve Stabia': 'Juve Stabia', 'Carrarese': 'Carrarese',
    'Manchester United': 'Man United', 'Manchester City': 'Man City', 'Tottenham Hotspur': 'Tottenham', 'Newcastle United': 'Newcastle', 'Wolverhampton Wanderers': 'Wolves', 'Brighton and Hove Albion': 'Brighton',
    'West Ham United': 'West Ham', 'Leeds United': 'Leeds', 'Leicester City': 'Leicester', 'Norwich City': 'Norwich', 'Sheffield United': 'Sheffield United', 'Blackburn Rovers': 'Blackburn', 
    'West Bromwich Albion': 'West Brom', 'Coventry City': 'Coventry', 'Middlesbrough': 'Middlesbrough', 'Stoke City': 'Stoke', 'Queens Park Rangers': 'QPR', 'Preston North End': 'Preston', 
    'Sheffield Wednesday': 'Sheffield Weds', 'Luton Town': 'Luton', 'Burnley': 'Burnley', 'Watford': 'Watford', 'Sunderland AFC': 'Sunderland', 'Sunderland': 'Sunderland',
    'Derby County': 'Derby', 'Birmingham City': 'Birmingham', 'Swansea City': 'Swansea', 'Wrexham AFC': 'Wrexham', 'Oxford United': 'Oxford', 'Charlton Athletic': 'Charlton',
    'Ipswich Town': 'Ipswich', 'Hull City': 'Hull', 'Bristol City': 'Bristol City', 'Cardiff City': 'Cardiff', 'Portsmouth': 'Portsmouth', 'Plymouth Argyle': 'Plymouth', 'Millwall': 'Millwall',
    'Nottingham Forest': "Nott'm Forest", 'Bolton Wanderers': 'Bolton', 'Bradford City': 'Bradford', 'Burton Albion': 'Burton', 'Doncaster Rovers': 'Doncaster', 'Exeter City': 'Exeter', 'Huddersfield Town': 'Huddersfield',
    'Lincoln City': 'Lincoln', 'Mansfield Town': 'Mansfield', 'Northampton Town': 'Northampton', 'Peterborough United': 'Peterboro', 'Rotherham United': 'Rotherham', 'Stockport County FC': 'Stockport',
    'Wigan Athletic': 'Wigan', 'Wimbledon': 'Wimbledon', 'Wycombe Wanderers': 'Wycombe', 'Bayern Munich': 'Bayern Munich', 'Bayer Leverkusen': 'Leverkusen', 'Borussia Dortmund': 'Dortmund',
    'Borussia Monchengladbach': "M'gladbach", '1. FC Köln': 'FC Koln', 'FSV Mainz 05': 'Mainz', 'Mainz 05': 'Mainz', 'VfL Wolfsburg': 'Wolfsburg', 'TSG Hoffenheim': 'Hoffenheim', 'Werder Bremen': 'Werder Bremen', 'Augsburg': 'Augsburg',
    '1. FC Heidenheim': 'Heidenheim', 'Hamburger SV': 'Hamburg', '1. FC Kaiserslautern': 'Kaiserslautern', '1. FC Magdeburg': 'Magdeburg', '1. FC Nürnberg': 'Nurnberg',
    'Arminia Bielefeld': 'Bielefeld', 'Dynamo Dresden': 'Dresden', 'Eintracht Braunschweig': 'Braunschweig', 'FC Schalke 04': 'Schalke 04', 'Fortuna Düsseldorf': 'Fortuna Dusseldorf', 'Greuther Fürth': 'Greuther Furth',
    'Hannover 96': 'Hannover', 'Hertha Berlin': 'Hertha', 'Karlsruher SC': 'Karlsruhe', 'SC Paderborn': 'Paderborn', 'SC Preußen Münster': 'Preussen Munster', 'SV Darmstadt 98': 'Darmstadt',
    'Eintracht Frankfurt': 'Ein Frankfurt', 'VfB Stuttgart': 'Stuttgart', 'SC Freiburg': 'Freiburg', 'Atletico Madrid': 'Ath Madrid', 'Athletic Bilbao': 'Ath Bilbao', 'Real Betis': 'Betis', 'Real Sociedad': 'Sociedad', 
    'Rayo Vallecano': 'Vallecano', 'Alavés': 'Alaves', 'Cadiz CF': 'Cadiz', 'UD Las Palmas': 'Las Palmas', 'Real Valladolid': 'Valladolid', 'Leganés': 'Leganes', 'Girona FC': 'Girona',
    'CA Osasuna': 'Osasuna', 'Elche CF': 'Elche', 'Celta Vigo': 'Celta', 'AD Ceuta FC': 'Ceuta', 'Almería': 'Almeria', 'Andorra CF': 'Andorra', 'Burgos CF': 'Burgos',
    'CD Castellón': 'Castellon', 'CD Mirandés': 'Mirandes', 'Cádiz CF': 'Cadiz', 'Córdoba': 'Cordoba', 'Deportivo La Coruña': 'La Coruna', 'Granada CF': 'Granada', 'Málaga': 'Malaga',
    'Real Racing Club de Santander': 'Santander', 'Real Valladolid CF': 'Valladolid', 'SD Eibar': 'Eibar', 'SD Huesca': 'Huesca', 'Sporting Gijón': 'Sp Gijon',
    'Paris Saint Germain': 'Paris SG', 'Marseille': 'Marseille', 'Lyon': 'Lyon', 'RC Lens': 'Lens', 'AS Monaco': 'Monaco', 'Lille OSC': 'Lille', 'Nice': 'Nice', 'Brest': 'Brest',
    'PSV Eindhoven': 'PSV Eindhoven', 'Feyenoord Rotterdam': 'Feyenoord', 'Ajax Amsterdam': 'Ajax', 'AZ Alkmaar': 'AZ Alkmaar', 'FC Twente': 'Twente', 'Sparta Rotterdam': 'Sparta Rotterdam', 
    'NEC Nijmegen': 'Nijmegen', 'Go Ahead Eagles': 'Go Ahead Eagles', 'Fortuna Sittard': 'For Sittard', 'PEC Zwolle': 'Zwolle', 'Almere City': 'Almere City', 'RKC Waalwijk': 'Waalwijk', 
    'SC Heerenveen': 'Heerenveen', 'Heracles Almelo': 'Heracles', 'FC Twente Enschede': 'Twente', 'FC Volendam': 'Volendam', 'FC Zwolle': 'Zwolle', 'SC Telstar': 'Telstar', 'FC Utrecht': 'Utrecht',
    'Benfica': 'Benfica', 'FC Porto': 'Porto', 'Vitoria Guimaraes': 'Guimaraes', 'Boavista FC': 'Boavista', 'Estoril Praia': 'Estoril', 'Casa Pia AC': 'Casa Pia',
    'Farense': 'Farense', 'Arouca': 'Arouca', 'Gil Vicente': 'Gil Vicente', 'AVS Futebol SAD': 'Avs', 'Braga': 'Sp Braga', 'SC Braga': 'Sp Braga', 'CF Estrela': 'Estrela',
    'Famalicão': 'Famalicao', 'Moreirense FC': 'Moreirense', 'Rio Ave FC': 'Rio Ave', 'Vitória SC': 'Guimaraes', 'Sporting CP': 'Sp Lisbon', 'Sporting Lisbon': 'Sp Lisbon',
    'Austria Wien': 'Austria Vienna', 'FC Blau-Weiß Linz': 'BW Linz', 'Grazer AK': 'Grazer', 'Hartberg': 'Hartberg', 'LASK': 'LASK Linz', 'RB Salzburg': 'Salzburg', 'Red Bull Salzburg': 'Salzburg',
    'Rapid Wien': 'Rapid Vienna', 'Rheindorf Altach': 'Altach', 'Ried': 'Ried', 'Sturm Graz': 'Sturm Graz', 'SK Sturm Graz': 'Sturm Graz', 'WSG Tirol': 'Tirol', 'Wolfsberger AC': 'Wolfsberger', 'Salzburg': 'Salzburg',
    'BSC Young Boys': 'Young Boys', 'Young Boys': 'Young Boys', 'FC Basel': 'Basel', 'FC Lausanne-Sport': 'Lausanne', 'FC Lugano': 'Lugano', 'Lugano': 'Lugano',
    'FC Luzern': 'Luzern', 'FC Sion': 'Sion', 'FC St Gallen': 'St Gallen', 'FC Thun': 'Thun', 'FC Winterthur': 'Winterthur', 'FC Zurich': 'Zurich', 'Grasshopper Zürich': 'Grasshoppers', 'Servette': 'Servette',
    'AE Kifisia FC': 'Kifisias', 'AEL': 'Larisa', 'Aris Thessaloniki': 'Aris', 'Atromitos Athens': 'Atromitos', 'Levadiakos': 'Levadiakos', 'PAOK Thessaloniki': 'PAOK', 'PAOK Salonika': 'PAOK',
    'Panetolikos Agrinio': 'Panetolikos', 'Panserraikos FC': 'Panserraikos', 'Volos FC': 'Volos NFC', 'Olympiakos Piraeus': 'Olympiakos', 'Panathinaikos FC': 'Panathinaikos', 'AEK Athens': 'AEK',
    'Basaksehir': 'Basaksehir', 'Istanbul Basaksehir': 'Basaksehir', 'Besiktas JK': 'Besiktas', 'Besiktas': 'Besiktas', 'Eyüpspor': 'Eyupspor', 'Fatih Karagümrük': 'Karagumruk',
    'Gazişehir Gaziantep': 'Gaziantep', 'Genclerbirligi SK': 'Genclerbirligi', 'Goztepe': 'Goztepe', 'Kasimpasa SK': 'Kasimpasa', 'Kasimpasa': 'Kasimpasa', 'Torku Konyaspor': 'Konyaspor', 'Çaykur Rizespor': 'Rizespor',
    'Galatasaray': 'Galatasaray', 'Fenerbahce': 'Fenerbahce', 'Trabzonspor': 'Trabzonspor', 'Celtic': 'Celtic', 'Rangers': 'Rangers', 'Rangers FC': 'Rangers', 'Aberdeen': 'Aberdeen', 'Hearts': 'Hearts',
    'KRC Genk': 'Genk', 'Union Saint-Gilloise': 'St Gilloise', 'AS Monaco': 'Monaco', 'AS Roma': 'Roma', 'Roma': 'Roma', 'Grimsby': 'Grimsby', 'Bodo/Glimt': 'Bodo/Glimt'
}

# ==============================================================================
# FUNZIONI CORE (REQUIRED)
# ==============================================================================

def parse_date(iso_str):
    try:
        dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ")
        dt_it = dt + timedelta(hours=1) 
        return dt_it.date(), dt_it.strftime("%d/%m %H:%M")
    except:
        return datetime.now().date(), "Oggi"

def load_odds_history():
    if os.path.exists(HISTORY_FILE): return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["MatchID", "Date", "Home", "Away", "Open_1", "Open_X", "Open_2", "Last_Update"])

def check_dropping_odds(h_team, a_team, date_str, c1, cX, c2):
    df_h = load_odds_history()
    m_id = f"{h_team}_{a_team}_{date_str}"
    row = df_h[df_h["MatchID"] == m_id]
    alert = ""
    if row.empty:
        new = pd.DataFrame([{"MatchID": m_id, "Date": date_str, "Home": h_team, "Away": a_team, "Open_1": c1, "Open_X": cX, "Open_2": c2, "Last_Update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
        df_h = pd.concat([df_h, new], ignore_index=True)
        df_h.to_csv(HISTORY_FILE, index=False)
    else:
        o1, o2 = float(row.iloc[0]["Open_1"]), float(row.iloc[0]["Open_2"])
        if o1 > 0 and c1 > 0 and ((o1-c1)/o1)*100 >= 10: alert += f"📉 DROP 1: {o1:.2f}->{c1:.2f} (-{((o1-c1)/o1)*100:.1f}%)\n"
        if o2 > 0 and c2 > 0 and ((o2-c2)/o2)*100 >= 10: alert += f"📉 DROP 2: {o2:.2f}->{c2:.2f} (-{((o2-c2)/o2)*100:.1f}%)\n"
    return alert

def calculate_elo_updates(df, league_code):
    base = 1500
    elo_dict = {}
    teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    for t in teams: elo_dict[t] = base
    for _, r in df.iterrows():
        h, a, res = r['HomeTeam'], r['AwayTeam'], r['Result']
        rh, ra = elo_dict[h], elo_dict[a]
        eh = 1/(1+10**((ra-rh)/400)); ea = 1/(1+10**((rh-ra)/400))
        sh, sa = (1,0) if res=='H' else (0,1) if res=='A' else (0.5,0.5)
        elo_dict[h] = rh + 32*(sh-eh); elo_dict[a] = ra + 32*(sa-ea)
    return elo_dict

@st.cache_data(ttl=3600)
def scarica_dati(lega):
    u1 = f"https://www.football-data.co.uk/mmz4281/{STAGIONE}/{lega}.csv"
    u2 = f"https://www.football-data.co.uk/new/{lega}.csv"
    try: df = pd.read_csv(u1)
    except: 
        try: df = pd.read_csv(u2)
        except: return None, None, None, None
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')
    df['HomeTeam'] = df['HomeTeam'].str.strip(); df['AwayTeam'] = df['AwayTeam'].str.strip()
    df['Result'] = np.where(df['FTHG'] > df['FTAG'], 'H', np.where(df['FTHG'] < df['FTAG'], 'A', 'D'))
    # Synthetic xG
    for c in ['HST','AST','HC','AC','HS','AS']: 
        if c not in df.columns: df[c] = 0.0
    df['xG_H'] = (df['HST']*0.32) + (np.maximum(0, df['HS']-df['HST'])*0.05) + (df['HC']*0.03)
    df['xG_A'] = (df['AST']*0.32) + (np.maximum(0, df['AS']-df['AST'])*0.05) + (df['AC']*0.03)
    elo = calculate_elo_updates(df, lega)
    # Averages
    avgs = {'Goals_H': df['FTHG'].mean(), 'Goals_A': df['FTAG'].mean(), 'xG_H': df['xG_H'].mean(), 'xG_A': df['xG_A'].mean()}
    h_df = df[['Date','HomeTeam','Result','FTHG','FTAG','xG_H','xG_A']].rename(columns={'HomeTeam':'Team','FTHG':'G_For','FTAG':'G_Ag','xG_H':'xG_F','xG_A':'xG_A_'})
    a_df = df[['Date','AwayTeam','Result','FTAG','FTHG','xG_A','xG_H']].rename(columns={'AwayTeam':'Team','FTAG':'G_For','FTHG':'G_Ag','xG_A':'xG_F','xG_H':'xG_A_'})
    full = pd.concat([h_df, a_df]).sort_values(['Team','Date'])
    # Weighted
    for m in ['G','xG']:
        att = 'G_For' if m=='G' else 'xG_F'
        full[f'W_{m}_A'] = full.groupby('Team')[att].transform(lambda x: x.ewm(span=5).mean())
        full[f'W_{m}_D'] = full.groupby('Team')['G_Ag' if m=='G' else 'xG_A_'].transform(lambda x: x.ewm(span=5).mean())
    return full, df, avgs, elo

def DixonColes(lh, la):
    mat = np.zeros((10,10))
    for i in range(10):
        for j in range(10): mat[i,j] = poisson.pmf(i,lh)*poisson.pmf(j,la)
    mat /= mat.sum()
    return 1/mat.sum(axis=0)[1:].sum(), 1/np.trace(mat), 1/mat.sum(axis=1)[1:].sum()

# ==============================================================================
# UI
# ==============================================================================

with st.sidebar:
    st.header("🎛️ Config")
    api = st.text_input("API Key", type="password")
    bank = st.number_input("Bank", value=26.5)
    codes = st.multiselect("Leghe", sorted(ALL_LEAGUES.keys()), default=['E0','I1'])

st.title("SmartBet Pro 60.0")

if st.button("🚀 ANALISI"):
    cache = {c: scarica_dati(c) for c in codes}
    for c in codes:
        url = f'https://api.the-odds-api.com/v4/sports/{API_MAPPING.get(c,"")}/odds/?apiKey={api}&regions={REGION}&markets={MARKET}'
        res = requests.get(url).json()
        if not isinstance(res, list): continue
        for m in res:
            h_r, a_r = m['home_team'], m['away_team']
            h_t, a_t = TEAM_MAPPING.get(h_r, h_r), TEAM_MAPPING.get(a_r, a_r)
            r_d, f_d = parse_date(m.get('commence_time',''))
            dfw, _, avgs, elo = cache[c]
            if dfw is None: continue
            hs, as_ = dfw[dfw['Team']==h_t], dfw[dfw['Team']==a_t]
            if hs.empty or as_.empty: continue
            # Calcolo Lambda
            lh = hs.iloc[-1]['W_G_A'] * as_.iloc[-1]['W_G_D'] / avgs['Goals_H']
            la = as_.iloc[-1]['W_G_A'] * hs.iloc[-1]['W_G_D'] / avgs['Goals_A']
            # Fatigue
            fat = ""
            dh, da = (r_d - hs.iloc[-1]['Date'].date()).days, (r_d - as_.iloc[-1]['Date'].date()).days
            if dh < 4: lh *= 0.85; fat += f"⚠️ STANCHEZZA {h_t} ({dh}gg)\n"
            if da < 4: la *= 0.85; fat += f"⚠️ STANCHEZZA {a_t} ({da}gg)\n"
            # Terminal
            st.markdown(f"""<div class='terminal-box'>
                <span class='term-header'>{h_t} vs {a_t} ({f_d})</span><br/>
                <span class='term-fatigue'>{fat}</span><br/>
                Exp Goals: {lh:.2f} - {la:.2f}
                </div>""", unsafe_allow_html=True)
