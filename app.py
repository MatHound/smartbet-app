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
st.set_page_config(page_title="SmartBet Pro 59.0 Live Fatigue", page_icon="⚡", layout="wide")

STAGIONE = "2526"
REGION = 'eu'
MARKET = 'h2h'
HISTORY_FILE = "smartbet_odds_history.csv" 

# CSS Custom (Terminal Style)
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
    .term-fatigue { color: #FFA500; font-weight: bold; border: 1px solid #FFA500; padding: 2px; }
    .term-drop { color: #00FF00; font-weight: bold; border: 1px solid #00FF00; padding: 2px; }
    .term-dim { color: #555555; }
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
    "📉 Leghe Minori (EU)": ['I2', 'E1', 'E2', 'E3', 'EC', 'D2', 'SP2', 'F2', 'SC1', 'SC2', 'SC3'],
    "🌎 Leghe Extra (Resto del Mondo)": ['ARG', 'BRA', 'CHN', 'DNK', 'FIN', 'IRL', 'JPN', 'MEX', 'NOR', 'POL', 'ROU', 'RUS', 'SWE', 'USA']
}
COMPACT_LEAGUES = LEAGUE_GROUPS["⚽ Europe Tier 2"] + LEAGUE_GROUPS["📉 Leghe Minori (EU)"] + LEAGUE_GROUPS["🌎 Leghe Extra (Resto del Mondo)"]

ALL_LEAGUES = {
    'UCL': '🇪🇺 Champions League', 'UEL': '🇪🇺 Europa League', 'UECL': '🇪🇺 Conference League',
    'I1': '🇮🇹 Serie A', 'E0': '🇬🇧 Premier League', 'SP1': '🇪🇸 La Liga', 'D1': '🇩🇪 Bundesliga', 'F1': '🇫🇷 Ligue 1',
    'N1': '🇳🇱 Eredivisie', 'P1': '🇵🇹 Primeira Liga', 'B1': '🇧🇪 Pro League', 'T1': '🇹🇷 Super Lig',
    'SC0': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Premiership', 'G1': '🇬🇷 Super League', 'A1': '🇦🇹 Bundesliga', 'SW1': '🇨🇭 Super League',
    'I2': '🇮🇹 Serie B', 'E1': '🇬🇧 Championship', 'E2': '🇬🇧 League One', 'E3': '🇬🇧 League Two', 'EC': '🇬🇧 National League',
    'D2': '🇩🇪 Bundesliga 2', 'SP2': '🇪🇸 Segunda', 'F2': '🇫🇷 Ligue 2',
    'SC1': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Championship', 'SC2': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 League One', 'SC3': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 League Two',
    'ARG': '🇦🇷 Argentina Primera', 'BRA': '🇧🇷 Brazil Serie A', 'CHN': '🇨🇳 China Super League',
    'DNK': '🇩🇰 Denmark Superliga', 'FIN': '🇫🇮 Finland Veikkausliiga', 'IRL': '🇮🇪 Ireland Premier',
    'JPN': '🇯🇵 Japan J-League', 'MEX': '🇲🇽 Mexico Liga MX', 'NOR': '🇳🇴 Norway Eliteserien',
    'POL': '🇵🇱 Poland Ekstraklasa', 'ROU': '🇷🇴 Romania Liga 1', 'RUS': '🇷🇺 Russia Premier',
    'SWE': '🇸🇪 Sweden Allsvenskan', 'USA': '🇺🇸 USA MLS'
}

API_MAPPING = {
    'UCL': 'soccer_uefa_champs_league', 'UEL': 'soccer_uefa_europa_league', 'UECL': 'soccer_uefa_euro_conference',
    'I1': 'soccer_italy_serie_a', 'I2': 'soccer_italy_serie_b', 'E0': 'soccer_epl', 'E1': 'soccer_efl_champ', 'E2': 'soccer_england_league1', 'E3': 'soccer_england_league2',
    'SP1': 'soccer_spain_la_liga', 'SP2': 'soccer_spain_segunda_division', 'D1': 'soccer_germany_bundesliga', 'D2': 'soccer_germany_bundesliga2',
    'F1': 'soccer_france_ligue_one', 'F2': 'soccer_france_ligue_two', 'N1': 'soccer_netherlands_eredivisie', 'P1': 'soccer_portugal_primeira_liga', 
    'B1': 'soccer_belgium_pro_league', 'T1': 'soccer_turkey_super_league', 'SC0': 'soccer_spfl_prem', 'G1': 'soccer_greece_super_league', 
    'A1': 'soccer_austria_bundesliga', 'SW1': 'soccer_switzerland_superleague', 'ARG': 'soccer_argentina_primera', 'BRA': 'soccer_brazil_campeonato', 
    'CHN': 'soccer_china_superleague', 'DNK': 'soccer_denmark_superliga', 'FIN': 'soccer_finland_veikkausliiga', 'IRL': 'soccer_ireland_premier_division',
    'JPN': 'soccer_japan_j_league', 'MEX': 'soccer_mexico_ligamx', 'NOR': 'soccer_norway_eliteserien', 'POL': 'soccer_poland_ekstraklasa', 'SWE': 'soccer_sweden_allsvenskan', 'USA': 'soccer_usa_mls'
}

LEAGUE_COEFF = {
    'E0': 1.00, 'SP1': 0.96, 'I1': 0.94, 'D1': 0.92, 'F1': 0.88, 'P1': 0.82, 'N1': 0.80, 'B1': 0.75, 'T1': 0.70, 'E1': 0.70, 'F2': 0.65,
    'SC0': 0.68, 'A1': 0.68, 'G1': 0.65, 'SW1': 0.65, 'D2': 0.65, 'I2': 0.60, 'SP2': 0.60, 'E2': 0.55, 'E3': 0.50, 'EC': 0.45,
    'SC1': 0.50, 'SC2': 0.45, 'SC3': 0.40, 'BRA': 0.75, 'ARG': 0.70, 'MEX': 0.65, 'USA': 0.65, 'JPN': 0.60
}

TEAM_MAPPING = {
    'Austria Wien': 'Austria Vienna', 'FC Blau-Weiß Linz': 'BW Linz', 'Grazer AK': 'GAK',
    'Hartberg': 'Hartberg', 'LASK': 'LASK Linz', 'RB Salzburg': 'Salzburg', 'Red Bull Salzburg': 'Salzburg',
    'Rapid Wien': 'Rapid Vienna', 'Rheindorf Altach': 'Altach', 'Ried': 'Ried',
    'Sturm Graz': 'Sturm Graz', 'SK Sturm Graz': 'Sturm Graz', 'WSG Tirol': 'Tirol', 'Wolfsberger AC': 'Wolfsberger',
    'Young Boys': 'Young Boys', 'Basel': 'Basel', '': 'Lausanne', 'Lugano': 'Lugano', 'Luzern': 'Luzern',
    'Sion': 'Sion', 'FC St Gallen': 'St Gallen', 'Thun': 'Thun', 'Winterthur': 'Winterthur', 'Zurich': 'Zurich', 'Grasshopper': 'Grasshoppers', 'Servette': 'Servette',
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
# ENGINE CORE FUNCTIONS
# ==============================================================================

def load_odds_history():
    if os.path.exists(HISTORY_FILE): return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["MatchID", "Date", "Home", "Away", "Open_1", "Open_X", "Open_2", "Last_Update"])

def save_odds_history(df): df.to_csv(HISTORY_FILE, index=False)

def check_dropping_odds(h_team, a_team, date_str, current_1, current_X, current_2):
    df_hist = load_odds_history()
    match_id = f"{h_team}_{a_team}_{date_str}"
    match_row = df_hist[df_hist["MatchID"] == match_id]
    drop_alert = ""
    if match_row.empty:
        new_row = pd.DataFrame([{"MatchID": match_id, "Date": date_str, "Home": h_team, "Away": a_team, "Open_1": current_1, "Open_X": current_X, "Open_2": current_2, "Last_Update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
        df_hist = pd.concat([df_hist, new_row], ignore_index=True)
        save_odds_history(df_hist)
    else:
        o1, o2 = float(match_row.iloc[0]["Open_1"]), float(match_row.iloc[0]["Open_2"])
        if o1 > 0 and current_1 > 0:
            d1 = ((o1 - current_1) / o1) * 100
            if d1 >= 10.0: drop_alert += f"📉 DROP 1: {o1:.2f}->{current_1:.2f} (-{d1:.1f}%)\n"
        if o2 > 0 and current_2 > 0:
            d2 = ((o2 - current_2) / o2) * 100
            if d2 >= 10.0: drop_alert += f"📉 DROP 2: {o2:.2f}->{current_2:.2f} (-{d2:.1f}%)\n"
        df_hist.loc[df_hist["MatchID"] == match_id, "Last_Update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_odds_history(df_hist)
    return drop_alert

def calculate_elo_updates(df_matches, league_code):
    base_rating = 1500; coeff = LEAGUE_COEFF.get(league_code, 0.60)
    if coeff >= 0.90: base_rating = 1600 
    elif coeff <= 0.55: base_rating = 1350
    elo_dict = {}; K = 32
    teams = set(df_matches['HomeTeam'].unique()) | set(df_matches['AwayTeam'].unique())
    for t in teams: elo_dict[t] = base_rating
    for _, row in df_matches.iterrows():
        h, a, res = row['HomeTeam'], row['AwayTeam'], row['Result']
        rh, ra = elo_dict[h], elo_dict[a]
        eh = 1 / (1 + 10 ** ((ra - rh) / 400)); ea = 1 / (1 + 10 ** ((rh - ra) / 400))
        sh, sa = (1.0, 0.0) if res == 'H' else (0.0, 1.0) if res == 'A' else (0.5, 0.5)
        elo_dict[h] = rh + K * (sh - eh); elo_dict[a] = ra + K * (sa - ea)
    return elo_dict

@st.cache_data(ttl=3600)
def scarica_dati(codice_lega):
    if codice_lega in ['UCL', 'UEL', 'UECL']: return None, None, None, None
    u1, u2 = f"https://www.football-data.co.uk/mmz4281/{STAGIONE}/{codice_lega}.csv", f"https://www.football-data.co.uk/new/{codice_lega}.csv"
    try: df = pd.read_csv(u1)
    except:
        try: df = pd.read_csv(u2)
        except: return None, None, None, None
    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date')
        df['HomeTeam'] = df['HomeTeam'].str.strip(); df['AwayTeam'] = df['AwayTeam'].str.strip()
        for col in ['HST','AST','HC','AC','HF','AF','HY','AY']:
            if col not in df.columns: df[col] = 0.0
        hst, hs, hc = df.get('HST', 0), df.get('HS', 0), df.get('HC', 0)
        ast, as_, ac = df.get('AST', 0), df.get('AS', 0), df.get('AC', 0)
        df['xG_H'] = (hst * 0.32) + (np.maximum(0, hs - hst) * 0.05) + (hc * 0.03)
        df['xG_A'] = (ast * 0.32) + (np.maximum(0, as_ - ast) * 0.05) + (ac * 0.03)
        df['Result'] = np.where(df['FTHG'] > df['FTAG'], 'H', np.where(df['FTHG'] < df['FTAG'], 'A', 'D'))
        elo = calculate_elo_updates(df, codice_lega)
        avgs = { 'Goals_H': df['FTHG'].mean(), 'Goals_A': df['FTAG'].mean(), 'xG_H': df['xG_H'].mean(), 'xG_A': df['xG_A'].mean(), 'Shots_H': df['HST'].mean(), 'Shots_A': df['AST'].mean(), 'Corn_H': df['HC'].mean(), 'Corn_A': df['AC'].mean(), 'Fouls_H': df['HF'].mean(), 'Fouls_A': df['AF'].mean(), 'Cards_H': df['HY'].mean(), 'Cards_A': df['AY'].mean() }
        h_df = df[['Date','HomeTeam','Result']].rename(columns={'HomeTeam':'Team'}); h_df['IsHome'] = 1; h_df['FormChar'] = np.where(h_df['Result'] == 'H', 'W', np.where(h_df['Result'] == 'A', 'L', 'D')); h_df['Goals_For'] = df['FTHG']; h_df['Goals_Ag'] = df['FTAG']; h_df['xG_For'] = df['xG_H']; h_df['xG_Ag'] = df['xG_A']; h_df['Shots_For'] = df['HST']; h_df['Shots_Ag'] = df['AST']; h_df['Corn_For'] = df['HC']; h_df['Corn_Ag'] = df['AC']; h_df['Fouls_For'] = df['HF']; h_df['Fouls_Ag'] = df['AF']; h_df['Cards_For'] = df['HY']; h_df['Cards_Ag'] = df['AY']
        a_df = df[['Date','AwayTeam','Result']].rename(columns={'AwayTeam':'Team'}); a_df['IsHome'] = 0; a_df['FormChar'] = np.where(a_df['Result'] == 'A', 'W', np.where(a_df['Result'] == 'H', 'L', 'D')); a_df['Goals_For'] = df['FTAG']; a_df['Goals_Ag'] = df['FTHG']; a_df['xG_For'] = df['xG_A']; a_df['xG_Ag'] = df['xG_H']; a_df['Shots_For'] = df['AST']; a_df['Shots_Ag'] = df['HST']; a_df['Corn_For'] = df['AC']; a_df['Corn_Ag'] = df['HC']; a_df['Fouls_For'] = df['AF']; a_df['Fouls_Ag'] = df['HF']; a_df['Cards_For'] = df['AY']; a_df['Cards_Ag'] = df['HY']
        full_df = pd.concat([h_df, a_df]).sort_values(['Team','Date'])
        for m in ['Goals', 'xG', 'Shots', 'Corn', 'Fouls', 'Cards']:
            full_df[f'{m}_Att_Rat'] = np.where(full_df['IsHome']==1, full_df[f'{m}_For']/avgs[f'{m}_H'], full_df[f'{m}_For']/avgs[f'{m}_A'])
            full_df[f'{m}_Def_Rat'] = np.where(full_df['IsHome']==1, full_df[f'{m}_Ag']/avgs[f'{m}_A'], full_df[f'{m}_Ag']/avgs[f'{m}_H'])
            full_df[f'W_{m}_Att'] = full_df.groupby('Team')[f'{m}_Att_Rat'].transform(lambda x: x.ewm(span=5, min_periods=1).mean())
            full_df[f'W_{m}_Def'] = full_df.groupby('Team')[f'{m}_Def_Rat'].transform(lambda x: x.ewm(span=5, min_periods=1).mean())
        return full_df, df, avgs, elo
    except: return None, None, None, None

def get_live_matches(api_key, sport_key):
    if not sport_key: return []
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions={REGION}&markets={MARKET}'
    try: return requests.get(url).json()
    except: return []

def calcola_1x2_dixon_coles(lam_h, lam_a):
    mat = np.zeros((10, 10))
    for i in range(10):
        for j in range(10): mat[i,j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
    rho = 0.13; mat[0,0] *= (1 - (lam_h * lam_a * rho)); mat[0,1] *= (1 + (lam_h * rho)); mat[1,0] *= (1 + (lam_a * rho)); mat[1,1] *= (1 - rho)
    mat /= np.sum(mat)
    p1, pX, p2 = np.sum(np.tril(mat,-1)), np.trace(mat), np.sum(np.triu(mat,1))
    return (1/p1 if p1>0 else 99), (1/pX if pX>0 else 99), (1/p2 if p2>0 else 99)

def find_team_stats_global(team_name, cache_dataframes):
    for code, (dfw, _, avgs, elo_dict) in cache_dataframes.items():
        if dfw is None: continue
        ts = dfw[dfw['Team'] == team_name]
        if not ts.empty:
            lr = ts.iloc[-1]; l5 = ts.tail(5)
            return lr, elo_dict.get(team_name, 1500), avgs, "-".join(l5['FormChar'].tolist()), code
    return None, 1500, None, "N/A", "N/A"

# ==============================================================================
# UI GENERATORS
# ==============================================================================

def generate_complete_terminal(h_team, a_team, exp, odds, roi, min_p, h_date, a_date, bank, h_form, a_form, code, h_elo, a_elo, fatigue, drop):
    html = f"<div class='terminal-box'><div class='term-header'>[ELO] {h_team} ({int(h_elo)}) vs {a_team} ({int(a_elo)})</div>"
    if h_elo - a_elo > 100: html += f"<span class='term-green'>>>> ELO FAV: {h_team} (+{int(h_elo-a_elo)})</span>\n"
    elif a_elo - h_elo > 100: html += f"<span class='term-green'>>>> ELO FAV: {a_team} (+{int(a_elo-h_elo)})</span>\n"
    if fatigue: html += f"<div class='term-fatigue'>{fatigue}</div>\n"
    if drop: html += f"<div class='term-drop'>{drop}</div>\n"
    html += f"FORMA: {h_team:<15} [{h_form}] vs [{a_form}] {a_team}\n\n<span class='term-section'>[ 1X2 & VALUE ]</span>\n"
    html += f"{'SEGNO':<6} | {'MY Q':<10} | {'BOOK':<8} | {'VALUE':<8} | {'STAKE'}\n" + "-"*60 + "\n"
    for s, r, b in [('1', roi['1'], odds['1']), ('X', roi['X'], odds['X']), ('2', roi['2'], odds['2'])]:
        mq = b/(r+1) if r+1>0 else 99; pr = 1/mq if mq>0 else 0
        stk = round(bank*((b*pr-(1-pr))/b)*0.3, 2) if r>0 else 0
        v_str = f"<span class='term-val'>{r*100:+.0f}% TOP</span>" if r>=0.15 else f"<span class='term-green'>{r*100:+.0f}%</span>" if r>0 else f"<span class='term-dim'>{r*100:+.0f}%</span>"
        html += f"{s:<6} | {mq:<10.2f} | {b:<8.2f} | {v_str:<20} | {f'€ {stk}' if stk>0 else '-'}\n"
    html += f"\n<span class='term-section'>[ ENGINE ]</span>\nCASA -> Goals: {exp['RealGoals'][0]:.2f} | xG: {exp['xG'][0]:.2f}\nOSP  -> Goals: {exp['RealGoals'][1]:.2f} | xG: {exp['xG'][1]:.2f}\n"
    html += "</div>"
    return html

# ==============================================================================
# MAIN INTERFACE
# ==============================================================================

with st.sidebar:
    st.header("🎛️ Config")
    api_key = st.text_input("API Key", type="password")
    bank = st.number_input("Bankroll (€)", value=26.50)
    min_p = st.slider("Min Prob", 0.50, 0.90, 0.65)
    st.divider()
    active_leagues = []
    for g, codes in LEAGUE_GROUPS.items():
        if st.checkbox(g, value=(g=="🏆 Top 5 (Tier 1)")): active_leagues.extend(codes)
    sel_codes = list(set(active_leagues + st.multiselect("Manuale:", sorted(ALL_LEAGUES.keys()))))

st.title("SmartBet Pro 59.0 Titanium Safe")
t1, t2 = st.tabs(["🚀 ANALISI", "📅 CAL"])

with t1:
    if st.button("🚀 AVVIA ANALISI"):
        st.session_state.results_data, cache = {}, {}
        for c in set([k for k in ALL_LEAGUES.keys() if k not in ['UCL','UEL','UECL']]): cache[c] = scarica_dati(c)
        for c in sel_codes:
            st.session_state.results_data[ALL_LEAGUES.get(c,c)] = []
            for m in get_live_matches(api_key, API_MAPPING.get(c,'')):
                h_r, a_r = m['home_team'], m['away_team']
                h_t, a_t = TEAM_MAPPING.get(h_r, h_r), TEAM_MAPPING.get(a_r, a_r)
                raw_d, fmt_d = parse_date(m.get('commence_time',''))
                h_st, h_el, h_av, h_f, h_lg = find_team_stats_global(h_t, cache)
                a_st, a_el, a_av, a_f, a_lg = find_team_stats_global(a_t, cache)
                if not h_st or not a_st: continue
                # Odds
                bks = m.get('bookmakers', [])
                q1, qX, q2 = [max([o['price'] for b in bks for mk in b['markets'] if mk['key']=='h2h' for o in mk['outcomes'] if o['name']==n] or [0]) for n in [h_r, 'Draw', a_r]]
                # Fatigue Live Fix
                mh, ma = (h_st['W_Goals_Att']*a_st['W_Goals_Def']*h_av['Goals_H']), (a_st['W_Goals_Att']*h_st['W_Goals_Def']*a_av['Goals_A'])
                xh, xa = (h_st['W_xG_Att']*a_st['W_xG_Def']*h_av['xG_H']), (a_st['W_xG_Att']*h_st['W_xG_Def']*a_av['xG_A'])
                bh, ba = (mh*0.5+xh*0.5), (ma*0.5+xa*0.5)
                if h_el-a_el>100: bh*=1.05
                elif a_el-h_el>100: ba*=1.05
                fat_str = ""
                # LIVE FATIGUE CHECK
                d_h, d_a = (raw_d - h_st['Date'].date()).days, (raw_d - a_st['Date'].date()).days
                if d_h < 4: bh*=0.85; fat_str += f"⚠️ STANCHEZZA CASA ({d_h}gg)\n"
                if d_a < 4: ba*=0.85; fat_str += f"⚠️ STANCHEZZA OSPITE ({d_a}gg)\n"
                # Results
                mq1, mqX, mq2 = calcola_1x2_dixon_coles(bh, ba)
                drp = check_dropping_odds(h_t, a_t, str(raw_d), q1, qX, q2)
                html = generate_complete_terminal(h_t, a_t, {'RealGoals':(mh,ma), 'xG':(xh,xa), 'Goals':(bh,ba)}, {'1':q1,'X':qX,'2':q2}, {'1':(q1/mq1-1),'X':(qX/mqX-1),'2':(q2/mq2-1)}, min_p, h_st['Date'], a_st['Date'], bank, h_f, a_f, c, h_el, a_el, fat_str, drp)
                st.session_state.results_data[ALL_LEAGUES.get(c,c)].append({'label':f"{fmt_d} | {h_t} vs {a_t}", 'html':html})
        st.rerun()

    if st.session_state.results_data:
        for l, ms in st.session_state.results_data.items():
            if ms:
                with st.expander(f"🏆 {l}"):
                    for m in ms:
                        with st.expander(m['label']): st.markdown(m['html'], unsafe_allow_html=True)
