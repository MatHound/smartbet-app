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
st.set_page_config(page_title="SmartBet Manager 51", page_icon="🛡️", layout="wide")

STAGIONE = "2526"
REGION = 'eu'
MARKET = 'h2h'
TRACKER_FILE = "smartbet_portfolio.csv"

# CSS Custom
st.markdown("""
<style>
    .stProgress { display: none; }
    .terminal-box { font-family: "Courier New", Courier, monospace; background-color: #0c0c0c; color: #cccccc; padding: 15px; border-radius: 5px; border: 1px solid #333; white-space: pre; overflow-x: auto; font-size: 0.9em; margin-bottom: 10px; }
    .terminal-missing { font-family: "Courier New", Courier, monospace; background-color: #1a1a1a; color: #777; padding: 15px; border-radius: 5px; border: 1px solid #550000; white-space: pre; overflow-x: auto; font-size: 0.9em; margin-bottom: 10px; }
    .term-header { color: #FFD700; font-weight: bold; } 
    .term-section { color: #00FFFF; font-weight: bold; margin-top: 10px; display: block; } 
    .term-green { color: #00FF00; font-weight: bold; } 
    .term-val { color: #FF00FF; font-weight: bold; } 
    .term-warn { color: #FF4500; font-weight: bold; background-color: #330000; padding: 2px; }
    .term-dim { color: #555555; }
    .streamlit-expanderHeader { font-weight: bold; background-color: #f0f2f6; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Session State
if 'results_data' not in st.session_state: st.session_state['results_data'] = {}
if 'calendar_data' not in st.session_state: st.session_state['calendar_data'] = []
if 'missing_log' not in st.session_state: st.session_state['missing_log'] = []

# --- DATABASE LEGHE ---
LEAGUE_GROUPS = {
    "🇪🇺 Coppe Europee": ['UCL', 'UEL', 'UECL'],
    "🏆 Top 5 (Tier 1)": ['I1', 'E0', 'SP1', 'D1', 'F1'],
    "⚽ Europe Tier 2": ['N1', 'P1', 'B1', 'T1', 'SC0', 'G1', 'A1', 'SW1'],
    "📉 Leghe Minori": ['I2', 'E1', 'E2', 'D2', 'SP2']
}

ALL_LEAGUES = {
    'UCL': '🇪🇺 Champions League', 'UEL': '🇪🇺 Europa League', 'UECL': '🇪🇺 Conference League',
    'I1': '🇮🇹 Serie A', 'E0': '🇬🇧 Premier League', 'SP1': '🇪🇸 La Liga', 'D1': '🇩🇪 Bundesliga', 'F1': '🇫🇷 Ligue 1',
    'N1': '🇳🇱 Eredivisie', 'P1': '🇵🇹 Primeira Liga', 'B1': '🇧🇪 Pro League', 'T1': '🇹🇷 Super Lig',
    'SC0': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Premiership', 'G1': '🇬🇷 Super League', 'A1': '🇦🇹 Bundesliga', 'SW1': '🇨🇭 Super League',
    'I2': '🇮🇹 Serie B', 'E1': '🇬🇧 Championship', 'E2': '🇬🇧 League One', 'D2': '🇩🇪 Bundesliga 2', 'SP2': '🇪🇸 Segunda'
}

API_MAPPING = {
    'UCL': 'soccer_uefa_champs_league', 'UEL': 'soccer_uefa_europa_league', 'UECL': 'soccer_uefa_euro_conference',
    'I1': 'soccer_italy_serie_a', 'I2': 'soccer_italy_serie_b',
    'E0': 'soccer_epl', 'E1': 'soccer_efl_champ', 'E2': 'soccer_england_league1',
    'SP1': 'soccer_spain_la_liga', 'SP2': 'soccer_spain_segunda_division',
    'D1': 'soccer_germany_bundesliga', 'D2': 'soccer_germany_bundesliga2',
    'F1': 'soccer_france_ligue_one', 
    'N1': 'soccer_netherlands_eredivisie', 'P1': 'soccer_portugal_primeira_liga', 
    'B1': 'soccer_belgium_pro_league', 'T1': 'soccer_turkey_super_league',
    'SC0': 'soccer_spfl_prem', 'G1': 'soccer_greece_super_league', 
    'A1': 'soccer_austria_bundesliga', 'SW1': 'soccer_switzerland_superleague'
}

LEAGUE_COEFF = {
    'E0': 1.00, 'SP1': 0.96, 'I1': 0.94, 'D1': 0.92, 'F1': 0.88,
    'P1': 0.82, 'N1': 0.80, 'B1': 0.75, 'T1': 0.70, 'E1': 0.70,
    'SC0': 0.68, 'A1': 0.68, 'G1': 0.65, 'SW1': 0.65,
    'D2': 0.65, 'I2': 0.60, 'SP2': 0.60, 'E2': 0.55
}

# --- MAPPING 46.3 ---
TEAM_MAPPING = {
    'Austria Wien': 'Austria Vienna', 'FC Blau-Weiß Linz': 'BW Linz', 'Grazer AK': 'GAK',
    'Hartberg': 'Hartberg', 'LASK': 'LASK Linz', 'RB Salzburg': 'Salzburg', 'Red Bull Salzburg': 'Salzburg',
    'Rapid Wien': 'Rapid Vienna', 'Rheindorf Altach': 'Altach', 'Ried': 'Ried',
    'Sturm Graz': 'Sturm Graz', 'SK Sturm Graz': 'Sturm Graz', 'WSG Tirol': 'Tirol', 'Wolfsberger AC': 'Wolfsberger',
    'BSC Young Boys': 'Young Boys', 'FC Basel': 'Basel', 'FC Lausanne-Sport': 'Lausanne',
    'FC Lugano': 'Lugano', 'Lugano': 'Lugano', 'FC Luzern': 'Luzern', 
    'FC Sion': 'Sion', 'FC St Gallen': 'St Gallen', 'FC Thun': 'Thun', 
    'FC Winterthur': 'Winterthur', 'FC Zurich': 'Zurich', 'Grasshopper Zürich': 'Grasshoppers', 'Servette': 'Servette',
    'Atlético Madrid': 'Ath Madrid', 'Espanyol': 'Espanyol', 'Real Sociedad B': 'Sociedad B',
    'Südtirol': 'Sudtirol', 'US Catanzaro 1929': 'Catanzaro',
    'SC Preußen Münster': 'Preussen Munster', 'VfL Bochum': 'Bochum', 
    'AE Kifisia FC': 'Kifisias', 'Levadiakos': 'Levadiakos',
    'AVS Futebol SAD': 'Avs', 'Wimbledon': 'AFC Wimbledon',
    'Basaksehir': 'Basaksehir', 'Goztepe': 'Goztepe',
    'Inter Milan': 'Inter', 'AC Milan': 'Milan', 'Napoli': 'Napoli', 'Juventus': 'Juventus',
    'Atalanta BC': 'Atalanta', 'Hellas Verona': 'Verona', 'Udinese Calcio': 'Udinese', 
    'Cagliari Calcio': 'Cagliari', 'US Lecce': 'Lecce', 'Empoli FC': 'Empoli', 
    'Sassuolo Calcio': 'Sassuolo', 'Salernitana': 'Salernitana', 'Monza': 'Monza', 
    'Frosinone': 'Frosinone', 'Genoa': 'Genoa', 'Parma': 'Parma', 'Como': 'Como', 
    'Venezia': 'Venezia', 'Pisa': 'Pisa', 'Cremonese': 'Cremonese', 'Palermo': 'Palermo', 
    'Bari': 'Bari', 'Sampdoria': 'Sampdoria', 'Spezia Calcio': 'Spezia',
    'Modena FC': 'Modena', 'Catanzaro': 'Catanzaro', 'Reggiana': 'Reggiana', 'Brescia': 'Brescia',
    'Cosenza': 'Cosenza', 'Sudtirol': 'Sudtirol', 'Cittadella': 'Cittadella', 'Mantova': 'Mantova',
    'Cesena FC': 'Cesena', 'Cesena': 'Cesena', 'Juve Stabia': 'Juve Stabia', 'Carrarese': 'Carrarese',
    'Manchester United': 'Man United', 'Manchester City': 'Man City', 'Tottenham Hotspur': 'Tottenham',
    'Newcastle United': 'Newcastle', 'Wolverhampton Wanderers': 'Wolves', 'Brighton and Hove Albion': 'Brighton',
    'West Ham United': 'West Ham', 'Leeds United': 'Leeds', 'Leicester City': 'Leicester', 
    'Norwich City': 'Norwich', 'Sheffield United': 'Sheffield United', 'Blackburn Rovers': 'Blackburn', 
    'West Bromwich Albion': 'West Brom', 'Coventry City': 'Coventry', 'Middlesbrough': 'Middlesbrough', 
    'Stoke City': 'Stoke', 'Queens Park Rangers': 'QPR', 'Preston North End': 'Preston', 
    'Sheffield Wednesday': 'Sheffield Weds', 'Luton Town': 'Luton', 'Burnley': 'Burnley', 
    'Watford': 'Watford', 'Sunderland AFC': 'Sunderland', 'Sunderland': 'Sunderland',
    'Derby County': 'Derby', 'Birmingham City': 'Birmingham', 'Swansea City': 'Swansea',
    'Wrexham AFC': 'Wrexham', 'Oxford United': 'Oxford', 'Charlton Athletic': 'Charlton',
    'Ipswich Town': 'Ipswich', 'Hull City': 'Hull', 'Bristol City': 'Bristol City', 
    'Cardiff City': 'Cardiff', 'Portsmouth': 'Portsmouth', 'Plymouth Argyle': 'Plymouth', 'Millwall': 'Millwall',
    'Nottingham Forest': "Nott'm Forest",
    'Bolton Wanderers': 'Bolton', 'Bradford City': 'Bradford', 'Burton Albion': 'Burton',
    'Doncaster Rovers': 'Doncaster', 'Exeter City': 'Exeter', 'Huddersfield Town': 'Huddersfield',
    'Lincoln City': 'Lincoln', 'Mansfield Town': 'Mansfield', 'Northampton Town': 'Northampton',
    'Peterborough United': 'Peterboro', 'Rotherham United': 'Rotherham', 'Stockport County FC': 'Stockport',
    'Wigan Athletic': 'Wigan', 'Wimbledon': 'Wimbledon', 'Wycombe Wanderers': 'Wycombe',
    'Bayern Munich': 'Bayern Munich', 'Bayer Leverkusen': 'Leverkusen', 'Borussia Dortmund': 'Dortmund',
    'Borussia Monchengladbach': "M'gladbach", '1. FC Köln': 'FC Koln', 'FSV Mainz 05': 'Mainz', 'Mainz 05': 'Mainz',
    'VfL Wolfsburg': 'Wolfsburg', 'FC St. Pauli': 'St Pauli', 'Holstein Kiel': 'Holstein Kiel',
    'TSG Hoffenheim': 'Hoffenheim', 'Werder Bremen': 'Werder Bremen', 'Augsburg': 'Augsburg',
    '1. FC Heidenheim': 'Heidenheim', 'Hamburger SV': 'Hamburg',
    '1. FC Kaiserslautern': 'Kaiserslautern', '1. FC Magdeburg': 'Magdeburg', '1. FC Nürnberg': 'Nurnberg',
    'Arminia Bielefeld': 'Bielefeld', 'Dynamo Dresden': 'Dresden', 'Eintracht Braunschweig': 'Braunschweig',
    'FC Schalke 04': 'Schalke 04', 'Fortuna Düsseldorf': 'Fortuna Dusseldorf', 'Greuther Fürth': 'Greuther Furth',
    'Hannover 96': 'Hannover', 'Hertha Berlin': 'Hertha', 'Karlsruher SC': 'Karlsruhe',
    'SC Paderborn': 'Paderborn', 'SC Preußen Münster': 'Preussen Munster', 'SV Darmstadt 98': 'Darmstadt',
    'Eintracht Frankfurt': 'Ein Frankfurt', 'VfB Stuttgart': 'Stuttgart', 'SC Freiburg': 'Freiburg',
    'Atletico Madrid': 'Ath Madrid', 'Athletic Bilbao': 'Ath Bilbao', 'Real Betis': 'Betis', 'Real Sociedad': 'Sociedad', 
    'Rayo Vallecano': 'Vallecano', 'Alavés': 'Alaves', 'Cadiz CF': 'Cadiz', 
    'UD Las Palmas': 'Las Palmas', 'RCD Espanyol': 'Espanyol', 'Espanyol': 'Espanyol',
    'Real Valladolid': 'Valladolid', 'Leganés': 'Leganes', 'Girona FC': 'Girona',
    'CA Osasuna': 'Osasuna', 'Elche CF': 'Elche', 'Celta Vigo': 'Celta',
    'AD Ceuta FC': 'Ceuta', 'Almería': 'Almeria', 'Andorra CF': 'Andorra', 'Burgos CF': 'Burgos',
    'CD Castellón': 'Castellon', 'CD Mirandés': 'Mirandes', 'Cádiz CF': 'Cadiz', 'Córdoba': 'Cordoba',
    'Deportivo La Coruña': 'La Coruna', 'Granada CF': 'Granada', 'Málaga': 'Malaga',
    'Real Racing Club de Santander': 'Santander', 'Real Sociedad B': 'R Sociedad B', 
    'Real Valladolid CF': 'Valladolid', 'SD Eibar': 'Eibar', 'SD Huesca': 'Huesca', 'Sporting Gijón': 'Sp Gijon',
    'Paris Saint Germain': 'Paris SG', 'Marseille': 'Marseille', 'Lyon': 'Lyon', 
    'RC Lens': 'Lens', 'AS Monaco': 'Monaco', 'Lille OSC': 'Lille', 'Nice': 'Nice', 'Brest': 'Brest',
    'PSV Eindhoven': 'PSV Eindhoven', 'Feyenoord Rotterdam': 'Feyenoord', 'Ajax Amsterdam': 'Ajax', 
    'AZ Alkmaar': 'AZ Alkmaar', 'FC Twente': 'Twente', 'Sparta Rotterdam': 'Sparta Rotterdam', 
    'NEC Nijmegen': 'Nijmegen', 'Go Ahead Eagles': 'Go Ahead Eagles', 'Fortuna Sittard': 'For Sittard', 
    'PEC Zwolle': 'Zwolle', 'Almere City': 'Almere City', 'RKC Waalwijk': 'Waalwijk', 
    'SC Heerenveen': 'Heerenveen', 'Heracles Almelo': 'Heracles',
    'FC Twente Enschede': 'Twente', 'FC Volendam': 'Volendam', 'FC Zwolle': 'Zwolle', 'SC Telstar': 'Telstar',
    'FC Utrecht': 'Utrecht',
    'Benfica': 'Benfica', 'FC Porto': 'Porto', 'Vitoria Guimaraes': 'Guimaraes',
    'Boavista FC': 'Boavista', 'Estoril Praia': 'Estoril', 'Casa Pia AC': 'Casa Pia',
    'Farense': 'Farense', 'Arouca': 'Arouca', 'Gil Vicente': 'Gil Vicente',
    'AVS Futebol SAD': 'Avs', 'Braga': 'Sp Braga', 'SC Braga': 'Sp Braga', 'CF Estrela': 'Estrela',
    'Famalicão': 'Famalicao', 'Moreirense FC': 'Moreirense', 'Rio Ave FC': 'Rio Ave',
    'Vitória SC': 'Guimaraes', 'Sporting CP': 'Sp Lisbon', 'Sporting Lisbon': 'Sp Lisbon',
    'Austria Wien': 'Austria Vienna', 'FC Blau-Weiß Linz': 'BW Linz', 'Grazer AK': 'Grazer',
    'Hartberg': 'Hartberg', 'LASK': 'LASK Linz', 'RB Salzburg': 'Salzburg', 'Red Bull Salzburg': 'Salzburg',
    'Rapid Wien': 'Rapid Vienna', 'Rheindorf Altach': 'Altach', 'Ried': 'Ried',
    'Sturm Graz': 'Sturm Graz', 'SK Sturm Graz': 'Sturm Graz', 'WSG Tirol': 'Tirol', 'Wolfsberger AC': 'Wolfsberger',
    'Salzburg': 'Salzburg',
    'BSC Young Boys': 'Young Boys', 'Young Boys': 'Young Boys', 'FC Basel': 'Basel',
    'FC Lausanne-Sport': 'Lausanne', 'FC Lugano': 'Lugano', 'Lugano': 'Lugano',
    'FC Luzern': 'Luzern', 'FC Sion': 'Sion', 'FC St Gallen': 'St Gallen',
    'FC Thun': 'Thun', 'FC Winterthur': 'Winterthur', 'FC Zurich': 'Zurich',
    'Grasshopper Zürich': 'Grasshoppers', 'Servette': 'Servette',
    'AE Kifisia FC': 'Kifisias', 'AEL': 'Larisa', 'Aris Thessaloniki': 'Aris',
    'Atromitos Athens': 'Atromitos', 'Levadiakos': 'Levadiakos', 
    'PAOK Thessaloniki': 'PAOK', 'PAOK Salonika': 'PAOK',
    'Panetolikos Agrinio': 'Panetolikos', 'Panserraikos FC': 'Panserraikos', 'Volos FC': 'Volos NFC',
    'Olympiakos Piraeus': 'Olympiakos', 'Panathinaikos FC': 'Panathinaikos', 'AEK Athens': 'AEK',
    'Basaksehir': 'Basaksehir', 'Istanbul Basaksehir': 'Basaksehir',
    'Besiktas JK': 'Besiktas', 'Besiktas': 'Besiktas',
    'Eyüpspor': 'Eyupspor', 'Fatih Karagümrük': 'Karagumruk',
    'Gazişehir Gaziantep': 'Gaziantep', 'Genclerbirligi SK': 'Genclerbirligi',
    'Goztepe': 'Goztepe', 'Kasimpasa SK': 'Kasimpasa', 'Kasimpasa': 'Kasimpasa',
    'Torku Konyaspor': 'Konyaspor', 'Çaykur Rizespor': 'Rizespor',
    'Galatasaray': 'Galatasaray', 'Fenerbahce': 'Fenerbahce', 'Trabzonspor': 'Trabzonspor',
    'Celtic': 'Celtic', 'Rangers': 'Rangers', 'Rangers FC': 'Rangers',
    'Aberdeen': 'Aberdeen', 'Hearts': 'Hearts',
    'KRC Genk': 'Genk', 'Union Saint-Gilloise': 'St Gilloise',
    'AS Monaco': 'Monaco', 'AS Roma': 'Roma', 'Roma': 'Roma'
}

# ==============================================================================
# FUNZIONI CORE
# ==============================================================================

def parse_date(iso_date_str):
    try:
        dt = datetime.strptime(iso_date_str, "%Y-%m-%dT%H:%M:%SZ")
        dt_ita = dt + timedelta(hours=1) 
        return dt_ita.date(), dt_ita.strftime("%d/%m %H:%M")
    except:
        return datetime.now().date(), "Oggi"

# --- TRACKER FUNCTIONS (MANAGED EDITION) ---
def load_portfolio():
    cols = ["Date", "League", "Match", "Bet", "Odds", "Stake", "Result", "Profit", "RawDate", "Pinned"]
    if os.path.exists(TRACKER_FILE):
        df = pd.read_csv(TRACKER_FILE)
        # Assicura compatibilità con vecchi file (aggiunge colonna Pinned se manca)
        if "Pinned" not in df.columns:
            df["Pinned"] = False
        
        # Clean duplicates (Shield)
        initial_len = len(df)
        df.drop_duplicates(subset=['Match', 'Bet', 'RawDate'], keep='last', inplace=True)
        if len(df) < initial_len: df.to_csv(TRACKER_FILE, index=False)
        return df
    
    return pd.DataFrame(columns=cols)

def save_bet_to_csv(date, league, match, bet, odds, stake, raw_date):
    df = load_portfolio()
    raw_date_str = str(raw_date)
    is_duplicate = not df[
        (df['Match'] == match) & (df['Bet'] == bet) & (df['RawDate'].astype(str) == raw_date_str)
    ].empty
    
    if is_duplicate: return False 
    
    new_row = pd.DataFrame([{
        "Date": date, "League": league, "Match": match, "Bet": bet, 
        "Odds": odds, "Stake": stake, "Result": "Pending", "Profit": 0.0, "RawDate": raw_date, "Pinned": False
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(TRACKER_FILE, index=False)
    return True

def check_results_automatic(portfolio_df, cache_dataframes):
    updated = False
    for idx, row in portfolio_df.iterrows():
        if row['Result'] != 'Pending': continue
        match_date = pd.to_datetime(row['RawDate']).date()
        if match_date > datetime.now().date(): continue
        
        h_team_bet = row['Match'].split(" vs ")[0]
        
        for code, data_tuple in cache_dataframes.items():
            if data_tuple is None: continue
            df_full, df_orig, _ = data_tuple
            match_row = df_orig[
                (df_orig['Date'].dt.date == match_date) & 
                ((df_orig['HomeTeam'] == TEAM_MAPPING.get(h_team_bet, h_team_bet)) | (df_orig['HomeTeam'] == h_team_bet))
            ]
            if not match_row.empty:
                res = match_row.iloc[0]
                if pd.isna(res['FTHG']): continue
                hg = int(res['FTHG']); ag = int(res['FTAG'])
                win = False
                bet_type = row['Bet']
                if "Ov 1.5" in bet_type and (hg+ag) > 1.5: win = True
                elif "Ov 2.5" in bet_type and (hg+ag) > 2.5: win = True
                elif "1" in bet_type and hg > ag: win = True
                elif "X" in bet_type and hg == ag: win = True
                elif "2" in bet_type and ag > hg: win = True
                elif "Gol" in bet_type and hg > 0 and ag > 0: win = True
                
                portfolio_df.at[idx, 'Result'] = "WIN" if win else "LOSS"
                portfolio_df.at[idx, 'Profit'] = (row['Stake'] * row['Odds'] - row['Stake']) if win else -row['Stake']
                updated = True
                break
    if updated: portfolio_df.to_csv(TRACKER_FILE, index=False)
    return portfolio_df

@st.cache_data(ttl=3600)
def scarica_dati(codice_lega):
    if codice_lega in ['UCL', 'UEL', 'UECL']: return None, None, None
    url = f"https://www.football-data.co.uk/mmz4281/{STAGIONE}/{codice_lega}.csv"
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date')
        df['HomeTeam'] = df['HomeTeam'].str.strip(); df['AwayTeam'] = df['AwayTeam'].str.strip()
        
        needed = ['Date','HomeTeam','AwayTeam','FTHG','FTAG']
        if not all(col in df.columns for col in needed): return None, None, None
        
        for col in ['HST','AST','HC','AC','HF','AF','HY','AY']:
            if col not in df.columns: df[col] = 0.0
            
        avgs = {
            'Goals_H': df['FTHG'].mean(), 'Goals_A': df['FTAG'].mean(),
            'Shots_H': df['HST'].mean(), 'Shots_A': df['AST'].mean(),
            'Corn_H': df['HC'].mean(), 'Corn_A': df['AC'].mean(),
            'Fouls_H': df['HF'].mean(), 'Fouls_A': df['AF'].mean(),
            'Cards_H': df['HY'].mean(), 'Cards_A': df['AY'].mean(),
        }
        
        df['Result'] = np.where(df['FTHG'] > df['FTAG'], 'H', np.where(df['FTHG'] < df['FTAG'], 'A', 'D'))
        
        h_df = df[['Date','HomeTeam','Result']].rename(columns={'HomeTeam':'Team'})
        h_df['IsHome'] = 1
        h_df['FormChar'] = np.where(h_df['Result'] == 'H', 'W', np.where(h_df['Result'] == 'A', 'L', 'D'))
        h_df['Goals_For'] = df['FTHG']; h_df['Shots_For'] = df['HST']; h_df['Corn_For'] = df['HC']; h_df['Fouls_For'] = df['HF']; h_df['Cards_For'] = df['HY']
        h_df['Goals_Ag'] = df['FTAG']; h_df['Shots_Ag'] = df['AST']; h_df['Corn_Ag'] = df['AC']; h_df['Fouls_Ag'] = df['AF']; h_df['Cards_Ag'] = df['AY']
        
        a_df = df[['Date','AwayTeam','Result']].rename(columns={'AwayTeam':'Team'})
        a_df['IsHome'] = 0
        a_df['FormChar'] = np.where(a_df['Result'] == 'A', 'W', np.where(a_df['Result'] == 'H', 'L', 'D'))
        a_df['Goals_For'] = df['FTAG']; a_df['Shots_For'] = df['AST']; a_df['Corn_For'] = df['AC']; a_df['Fouls_For'] = df['AF']; a_df['Cards_For'] = df['AY']
        a_df['Goals_Ag'] = df['FTHG']; a_df['Shots_Ag'] = df['HST']; a_df['Corn_Ag'] = df['HC']; a_df['Fouls_Ag'] = df['HF']; a_df['Cards_Ag'] = df['HY']
        
        full_df = pd.concat([h_df, a_df]).sort_values(['Team','Date'])
        
        metrics = ['Goals', 'Shots', 'Corn', 'Fouls', 'Cards']
        for m in metrics:
            full_df[f'{m}_Att_Rat'] = np.where(full_df['IsHome']==1, full_df[f'{m}_For']/avgs[f'{m}_H'], full_df[f'{m}_For']/avgs[f'{m}_A'])
            full_df[f'{m}_Def_Rat'] = np.where(full_df['IsHome']==1, full_df[f'{m}_Ag']/avgs[f'{m}_A'], full_df[f'{m}_Ag']/avgs[f'{m}_H'])
            
            full_df[f'W_{m}_Att'] = full_df.groupby('Team')[f'{m}_Att_Rat'].transform(lambda x: x.ewm(span=5, min_periods=1).mean())
            full_df[f'W_{m}_Def'] = full_df.groupby('Team')[f'{m}_Def_Rat'].transform(lambda x: x.ewm(span=5, min_periods=1).mean())

        return full_df, df, avgs
    except: return None, None, None

def get_live_matches(api_key, sport_key):
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions={REGION}&markets={MARKET}'
    try: return requests.get(url).json()
    except: return []

def calculate_kelly_stake(bankroll, odds, probability, fraction=0.3):
    if odds <= 1 or probability <= 0: return 0.0
    b = odds - 1; q = 1 - probability
    f = (b * probability - q) / b
    return round(bankroll * max(0, f) * fraction, 2)

def calcola_1x2_dixon_coles(lam_h, lam_a):
    mat = np.zeros((6,6))
    for i in range(6):
        for j in range(6): mat[i,j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
    rho = 0.13
    mat[0,0] *= (1 - (lam_h * lam_a * rho)); mat[0,1] *= (1 + (lam_h * rho))
    mat[1,0] *= (1 + (lam_a * rho)); mat[1,1] *= (1 - rho)
    mat = mat / np.sum(mat)
    p1 = np.sum(np.tril(mat,-1)); pX = np.trace(mat); p2 = np.sum(np.triu(mat,1))
    return (1/p1 if p1>0 else 99), (1/pX if pX>0 else 99), (1/p2 if p2>0 else 99)

def calcola_h2h_favorito(val_h, val_a):
    r = np.arange(40)
    pmf_h = poisson.pmf(r, val_h); pmf_a = poisson.pmf(r, val_a)
    joint = np.outer(pmf_h, pmf_a)
    p_h = np.sum(np.tril(joint, -1)); p_a = np.sum(np.triu(joint, 1))
    return p_h, p_a

def find_team_stats_global(team_name, cache_dataframes):
    for league_code, (df_weighted, _, averages) in cache_dataframes.items():
        if df_weighted is None: continue
        team_stats = df_weighted[df_weighted['Team'] == team_name]
        if not team_stats.empty:
            last_row = team_stats.iloc[-1]
            last_5 = df_weighted[df_weighted['Team'] == team_name].tail(5)
            form_str = "-".join(last_5['FormChar'].tolist())
            coeff = LEAGUE_COEFF.get(league_code, 0.65)
            return last_row, coeff, averages, form_str, league_code
    return None, 0, None, "N/A", "N/A"

def generate_missing_data_terminal(h_team, a_team, h_found, a_found, bookie_odds):
    html = f"""<div class='terminal-missing'>"""
    html += f"<span style='color:#FF5555; font-weight:bold;'>[ ! ] DATI INSUFFICIENTI: {h_team} vs {a_team}</span>\n"
    if not h_found: html += f"❌ Missing: {h_team}\n"
    if not a_found: html += f"❌ Missing: {a_team}\n"
    html += f"\nODDS: 1:{bookie_odds['1']:.2f} X:{bookie_odds['X']:.2f} 2:{bookie_odds['2']:.2f}</div>"
    return html

def generate_complete_terminal(h_team, a_team, exp_data, odds_1x2, roi_1x2, min_prob, last_date_h, last_date_a, bankroll, h_form, a_form):
    html = f"""<div class='terminal-box'>"""
    max_date = max(last_date_h, last_date_a)
    days_lag = (datetime.now() - max_date).days
    if days_lag > 14: html += f"<div class='term-warn'>⚠️ DATI VECCHI ({days_lag}gg)</div>\n"
    html += f"FORMA: {h_team:<15} [{h_form}] vs [{a_form}] {a_team}\n"
    
    # 1X2
    html += f"\n<span class='term-section'>[ 1X2 & MONEY MANAGEMENT ]</span>\n"
    html += f"{'SEGNO':<6} | {'MY QUOTA':<10} | {'BOOKIE':<8} | {'VALUE':<8} | {'PUNTA €'}\n"
    html += "-"*60 + "\n"
    segni = [('1', roi_1x2['1'], odds_1x2['1']), ('X', roi_1x2['X'], odds_1x2['X']), ('2', roi_1x2['2'], odds_1x2['2'])]
    for segno, roi, book_q in segni:
        my_q = book_q / (roi + 1) if (roi+1) > 0 else 99.0
        prob = 1/my_q if my_q > 0 else 0
        stake = calculate_kelly_stake(bankroll, book_q, prob) if roi > 0 else 0.0
        
        if book_q == 0: val_str="ERR"
        elif roi >= 0.15 and book_q <= 4.0 and prob >= min_prob: val_str = f"<span class='term-val'>{roi*100:+.0f}% (TOP)</span>"
        elif roi > 0 and prob >= min_prob: val_str = f"<span class='term-green'>{roi*100:+.0f}%</span>"
        else: val_str = f"<span class='term-dim'>{roi*100:+.0f}%</span>"
        
        stake_str = f"<span class='term-money'>€ {stake:.2f}</span>" if stake > 0 else "-"
        html += f"{segno:<6} | {my_q:<10.2f} | {book_q:<8.2f} | {val_str:<20} | {stake_str}\n"

    # TESTA A TESTA
    html += f"\n<span class='term-section'>[ TESTA A TESTA ]</span>\n"
    metrics_cfg = [("GOL", 'Goals'), ("CORNER", 'Corn'), ("TIRI", 'Shots'), ("FALLI", 'Fouls'), ("CARDS", 'Cards')]
    for label, key in metrics_cfg:
        ph, pa = calcola_h2h_favorito(exp_data[key][0], exp_data[key][1])
        if ph > pa:
            fav_str = f"CASA ({ph*100:.0f}%)"
            if ph >= min_prob: fav_str = f"<span class='term-green'>{fav_str}</span>"
            else: fav_str = f"<span class='term-dim'>{fav_str}</span>"
        else:
            fav_str = f"OSP ({pa*100:.0f}%)"
            if pa >= min_prob: fav_str = f"<span class='term-green'>{fav_str}</span>"
            else: fav_str = f"<span class='term-dim'>{fav_str}</span>"
        val_h, val_a = exp_data[key]
        html += f"{label:<10}: {fav_str}  [Exp: {val_h:.1f} vs {val_a:.1f}]\n"

    # DETTAGLIO PROP
    prop_configs = [
        ("GOL", exp_data['Goals'], [0.5, 1.5, 2.5], [0.5, 1.5], [1.5, 2.5, 3.5]),
        ("CORNER", exp_data['Corn'], [3.5, 4.5, 5.5], [2.5, 3.5, 4.5], [8.5, 9.5, 10.5]),
        ("TIRI PORTA", exp_data['Shots'], [3.5, 4.5, 5.5], [2.5, 3.5, 4.5], [7.5, 8.5, 9.5]),
        ("FALLI", exp_data['Fouls'], [10.5, 11.5, 12.5], [10.5, 11.5, 12.5], [21.5, 22.5, 23.5]),
        ("CARDS", exp_data['Cards'], [1.5, 2.5], [1.5, 2.5], [3.5, 4.5])
    ]
    for label, (eh, ea), r_h, r_a, r_tot in prop_configs:
        html += f"\n<span class='term-section'>[ {label} DETTAGLIO ]</span>\n"
        html += f"{'LINEA':<15} | {'PROB %':<8} | {'QUOTA'}\n"
        html += "-"*40 + "\n"
        def add_rows(prefix, r, exp):
            rows_html = ""
            for l in r:
                p = poisson.sf(int(l), exp)
                q = 1/p if p > 0 else 99
                row_str = f"{prefix+' Ov '+str(l):<15} | {p*100:04.1f}%   | {q:.2f}"
                if p >= min_prob: rows_html += f"<span class='term-green'>{row_str}</span>\n"
                else: rows_html += f"<span class='term-dim'>{row_str}</span>\n"
            return rows_html
        html += add_rows("CASA", r_h, eh)
        html += add_rows("OSP", r_a, ea)
        html += add_rows("TOT", r_tot, eh+ea)

    html += "</div>"
    return html

# ==============================================================================
# INTERFACCIA
# ==============================================================================

with st.sidebar:
    st.header("🎛️ Configurazione")
    api_key_input = st.text_input("API Key", type="password")
    bankroll_input = st.number_input("Bankroll (€)", min_value=10.0, value=26.50, step=0.5)
    
    st.divider()
    min_prob_val = st.slider("Probabilità Minima (Verde)", 0.50, 0.90, 0.65, step=0.05)
    
    st.divider()
    st.markdown("### 🏆 Campionati")
    
    active_groups = []
    col1, col2 = st.columns(2)
    for idx, (g_name, g_codes) in enumerate(LEAGUE_GROUPS.items()):
        with col1 if idx % 2 == 0 else col2:
            if st.checkbox(g_name, value=(g_name == "🇪🇺 Coppe Europee")):
                active_groups.extend(g_codes)
                
    st.markdown("#### 2️⃣ Selezione Manuale")
    all_league_options = sorted(list(ALL_LEAGUES.keys()))
    manual_selection = st.multiselect(
        "Aggiungi Leghe:",
        options=all_league_options,
        format_func=lambda x: f"{ALL_LEAGUES[x]} ({x})",
        default=[]
    )
    final_selection_codes = list(set(active_groups + manual_selection))
    st.caption(f"Totale leghe selezionate: {len(final_selection_codes)}")
    
    st.divider()
    show_mapping_errors = st.checkbox("🛠️ Debug Mapping", value=False)
    inspect_csv_mode = st.checkbox("🔍 ISPEZIONA NOMI CSV", value=False)

st.title("SmartBet Manager 51")
st.caption("Editor Interattivo: Fissa (📌) e Cancella (❌)")

# TABS PRINCIPALI
tab_main, tab_cal, tab_tracker = st.tabs(["🚀 ANALISI MATCH", "📅 CALENDARIO", "💰 REGISTRO"])

with tab_main:
    start_analisys = st.button("🚀 CERCA VALUE BETS", type="primary", use_container_width=True)

    if inspect_csv_mode and api_key_input and final_selection_codes:
        st.info("MODALITÀ ISPEZIONE ATTIVA...")
        domestic_cache = {}
        leagues_to_load = [k for k in final_selection_codes if k not in ['UCL','UEL','UECL']]
        if any(c in ['UCL','UEL','UECL'] for c in final_selection_codes):
            leagues_to_load = [k for k in ALL_LEAGUES.keys() if k not in ['UCL','UEL','UECL']]
        for code in leagues_to_load:
            df, _, _ = scarica_dati(code)
            if df is not None:
                teams = sorted(df['Team'].unique())
                with st.expander(f"Squadre in {ALL_LEAGUES[code]} ({code})"):
                    st.code("\n".join(teams))
        st.stop() 

    if start_analisys:
        if not api_key_input: st.error("Inserisci API Key!")
        elif not final_selection_codes: st.error("Seleziona almeno una lega!")
        else:
            # RESET SESSION
            st.session_state['results_data'] = {}
            st.session_state['calendar_data'] = []
            st.session_state['missing_log'] = []
            
            domestic_cache = {}
            has_cups = any(c in ['UCL','UEL','UECL'] for c in final_selection_codes)
            leagues_to_load = [k for k in ALL_LEAGUES.keys() if k not in ['UCL','UEL','UECL']] if has_cups else [k for k in final_selection_codes if k not in ['UCL','UEL','UECL']]
            
            status = st.empty()
            status.text("Caricamento database statistici...")
            for idx, code in enumerate(leagues_to_load): domestic_cache[code] = scarica_dati(code)
                
            progress = st.progress(0)
            total_steps = len(final_selection_codes)
            
            for idx, code in enumerate(final_selection_codes):
                progress.progress((idx+1)/total_steps)
                league_name = ALL_LEAGUES.get(code, code)
                status.text(f"Analisi: {league_name}...")
                
                if league_name not in st.session_state['results_data']:
                    st.session_state['results_data'][league_name] = []
                
                matches = get_live_matches(api_key_input, API_MAPPING.get(code, ''))
                
                if matches:
                    for m in matches:
                        if 'home_team' not in m: continue
                        h_raw, a_raw = m['home_team'], m['away_team']
                        h_team = TEAM_MAPPING.get(h_raw, h_raw)
                        a_team = TEAM_MAPPING.get(a_raw, a_raw)
                        raw_date_obj, fmt_date_str = parse_date(m.get('commence_time', ''))
                        
                        h_data, h_coeff, h_avgs, h_form, h_lg = find_team_stats_global(h_team, domestic_cache)
                        a_data, a_coeff, a_avgs, a_form, a_lg = find_team_stats_global(a_team, domestic_cache)
                        
                        q1_b, qX_b, q2_b = 0,0,0
                        for b in m['bookmakers']:
                            for mk in b['markets']:
                                if mk['key'] == 'h2h':
                                    for o in mk['outcomes']:
                                        if o['name'] == h_raw: q1_b = o['price']
                                        elif o['name'] == 'Draw': qX_b = o['price']
                                        elif o['name'] == a_raw: q2_b = o['price']
                        
                        if h_data is None or a_data is None:
                            if h_data is None: st.session_state['missing_log'].append(f"LEGA {code}: '{h_raw}' -> Missing")
                            if a_data is None: st.session_state['missing_log'].append(f"LEGA {code}: '{a_raw}' -> Missing")
                            html_err = generate_missing_data_terminal(h_team, a_team, (h_data is not None), (a_data is not None), {'1':q1_b,'X':qX_b,'2':q2_b})
                            item_err = {'label': f"⚠️ {fmt_date_str} | {h_team} vs {a_team} ({code})", 'html': html_err, 'raw_date': raw_date_obj}
                            st.session_state['results_data'][league_name].append(item_err)
                            st.session_state['calendar_data'].append(item_err)
                            continue

                        # Matrix
                        exp_data = {} 
                        metrics = ['Goals', 'Shots', 'Corn', 'Fouls', 'Cards']
                        for met in metrics:
                            h_att_r = h_data[f'W_{met}_Att']; h_def_r = h_data[f'W_{met}_Def']; h_lea_avg_h = h_avgs[f'{met}_H']
                            a_att_r = a_data[f'W_{met}_Att']; a_def_r = a_data[f'W_{met}_Def']; a_lea_avg_a = a_avgs[f'{met}_A']
                            
                            f_h_coeff = h_coeff if code in ['UCL', 'UEL', 'UECL'] else 1.0
                            f_a_coeff = a_coeff if code in ['UCL', 'UEL', 'UECL'] else 1.0
                            
                            val_h = h_att_r * a_def_r * h_lea_avg_h * f_h_coeff
                            val_a = a_att_r * h_def_r * a_lea_avg_a * f_a_coeff
                            exp_data[met] = (val_h, val_a)

                        if q1_b == 0: continue
                        my_q1, my_qX, my_q2 = calcola_1x2_dixon_coles(exp_data['Goals'][0], exp_data['Goals'][1])
                        roi_1 = ((1/my_q1)*q1_b)-1; roi_X = ((1/my_qX)*qX_b)-1; roi_2 = ((1/my_q2)*q2_b)-1
                        
                        # --- AUTO SNIPER LOGIC ---
                        segni_check = [('1', roi_1, q1_b, 1/my_q1), ('X', roi_X, qX_b, 1/my_qX), ('2', roi_2, q2_b, 1/my_q2)]
                        for s_lbl, s_roi, s_odd, s_prob in segni_check:
                            if s_roi >= 0.15 and s_odd <= 4.0 and s_prob >= min_prob_val:
                                stake = calculate_kelly_stake(bankroll_input, s_odd, s_prob)
                                save_bet_to_csv(raw_date_obj, league_name, f"{h_team} vs {a_team}", s_lbl, s_odd, stake, raw_date_obj)
                        # -------------------------

                        html_block = generate_complete_terminal(
                            h_team, a_team, exp_data, 
                            {'1':q1_b,'X':qX_b,'2':q2_b}, {'1':roi_1,'X':roi_X,'2':roi_2},
                            min_prob_val, h_data['Date'], a_data['Date'], bankroll_input, h_form, a_form
                        )
                        
                        item_ok = {'label': f"✅ {fmt_date_str} | {h_team} vs {a_team} ({code})", 'html': html_block, 'raw_date': raw_date_obj}
                        st.session_state['results_data'][league_name].append(item_ok)
                        st.session_state['calendar_data'].append(item_ok)

            status.empty()
            st.success("Analisi Completata. Le giocate TOP sono state salvate nel Registro!")
            
            if show_mapping_errors and st.session_state['missing_log']:
                st.warning(f"⚠️ Debug: {len(st.session_state['missing_log'])} squadre non trovate.")
                st.text_area("📋 Copia questa lista:", value="\n".join(sorted(list(set(st.session_state['missing_log'])))), height=200)

    if st.session_state['results_data']:
        active_leagues = [l for l in st.session_state['results_data'] if st.session_state['results_data'][l]]
        if active_leagues:
            tabs = st.tabs(active_leagues)
            for i, l in enumerate(active_leagues):
                with tabs[i]:
                    for m in st.session_state['results_data'][l]:
                        with st.expander(m['label']): st.markdown(m['html'], unsafe_allow_html=True)
        else: st.write("Nessun risultato.")

# TAB CALENDARIO
with tab_cal:
    st.markdown("### 📅 Calendario")
    if st.session_state['calendar_data']:
        unique_dates = sorted(list(set([x['raw_date'] for x in st.session_state['calendar_data']])))
        if unique_dates:
            selected_date = st.date_input("Seleziona Data:", value=unique_dates[0], min_value=unique_dates[0])
            daily_matches = [x for x in st.session_state['calendar_data'] if x['raw_date'] == selected_date]
            if daily_matches:
                for dm in daily_matches:
                    with st.expander(dm['label']): st.markdown(dm['html'], unsafe_allow_html=True)
            else: st.warning("Nessuna partita.")
    else: st.info("Nessun dato.")

# TAB TRACKER (MANAGER MODE)
with tab_tracker:
    st.markdown("### 📊 Registro & Manager")
    
    # UPLOAD/DOWNLOAD
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(TRACKER_FILE):
            with open(TRACKER_FILE, "rb") as f:
                st.download_button("📥 SCARICA BACKUP", f, file_name="smartbet_portfolio.csv", mime="text/csv")
    with c2:
        uploaded_file = st.file_uploader("📂 RIPRISTINA", type="csv")
        if uploaded_file is not None:
            pd.read_csv(uploaded_file).to_csv(TRACKER_FILE, index=False)
            st.success("Ripristinato!")

    pf = load_portfolio()
    if pf.empty:
        st.info("Nessuna giocata.")
    else:
        # DATA EDITOR
        st.markdown("#### Modifica Giocate")
        # Add temporary column for deletion in the UI (not saved in CSV yet)
        if "Delete" not in pf.columns:
            pf["Delete"] = False
            
        edited_pf = st.data_editor(
            pf,
            column_config={
                "Pinned": st.column_config.CheckboxColumn("📌 Fix", help="Se attivo, questa riga non verrà mai cancellata", default=False),
                "Delete": st.column_config.CheckboxColumn("❌ Del", help="Seleziona per eliminare", default=False),
                "Result": st.column_config.SelectboxColumn("Result", options=["WIN", "LOSS", "Pending"]),
            },
            disabled=["Date", "League", "Match", "Bet", "Odds", "Stake", "Profit"],
            hide_index=True,
            use_container_width=True
        )
        
        c_save, c_refresh = st.columns(2)
        
        if c_save.button("💾 APPLICA MODIFICHE (Salva/Cancella)"):
            # Logic: Keep rows where (Delete is False) OR (Pinned is True)
            # Pinned protects from deletion
            final_df = edited_pf[ (~edited_pf['Delete']) | (edited_pf['Pinned']) ].copy()
            final_df.drop(columns=['Delete'], inplace=True, errors='ignore')
            final_df.to_csv(TRACKER_FILE, index=False)
            st.success("Registro aggiornato!")
            st.rerun()
            
        if c_refresh.button("🔄 AUTO-GRADING (Controlla Risultati)"):
            with st.spinner("Controllo..."):
                domestic_cache = {}
                for k in ALL_LEAGUES.keys(): 
                    if k not in ['UCL','UEL','UECL']: domestic_cache[k] = scarica_dati(k)
                
                # Reload from file to ensure we use clean data
                current_df = load_portfolio()
                updated_df = check_results_automatic(current_df, domestic_cache)
                st.success("Fatto!")
                st.rerun()

        # METRICS
        wins = len(pf[pf['Result'] == 'WIN']); losses = len(pf[pf['Result'] == 'LOSS'])
        profit = pf['Profit'].sum(); roi = (profit / pf['Stake'].sum() * 100) if pf['Stake'].sum() > 0 else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Giocate", len(pf)); m2.metric("W/L", f"{wins}/{losses}")
        m3.metric("Profitto", f"€ {profit:.2f}", delta_color="normal")
        m4.metric("ROI", f"{roi:.1f}%")
        
        if st.button("🗑️ RESET TOTALE (Salva solo i Pinned)"):
            if os.path.exists(TRACKER_FILE):
                # Load, filter for Pinned, Save
                df_reset = pd.read_csv(TRACKER_FILE)
                if "Pinned" in df_reset.columns:
                    df_reset = df_reset[df_reset['Pinned'] == True]
                else:
                    df_reset = df_reset.iloc[0:0] # Empty
                
                df_reset.to_csv(TRACKER_FILE, index=False)
                st.warning("Reset effettuato (Righe fissate mantenute).")
                st.rerun()
