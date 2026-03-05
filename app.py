import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from scipy.stats import poisson
from datetime import datetime, timedelta
import google.generativeai as genai 

# ==============================================================================
# 1. CONFIGURAZIONE
# ==============================================================================
st.set_page_config(page_title="SmartBet Pro 63.1 Risk Manager", page_icon="🧬", layout="wide")

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
    .ai-box { background-color: #1a233a; border-left: 4px solid #00FFFF; padding: 10px; font-family: "Courier New", Courier, monospace; color: #e0e0e0; margin-top: 10px; border-radius: 3px; }
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
    "📉 Leghe Minori (EU)": ['I2', 'E1', 'E2', 'E3', 'EC', 'D2', 'SP2', 'F2', 'SC1', 'SC2', 'SC3'],
    "🌎 Leghe Extra (Resto del Mondo)": ['ARG', 'BRA', 'CHN', 'DNK', 'FIN', 'IRL', 'JPN', 'MEX', 'NOR', 'POL', 'ROU', 'RUS', 'SWE', 'USA']
}

TOP_5_LEAGUES = LEAGUE_GROUPS["🏆 Top 5 (Tier 1)"]
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
    'I1': 'soccer_italy_serie_a', 'I2': 'soccer_italy_serie_b',
    'E0': 'soccer_epl', 'E1': 'soccer_efl_champ', 'E2': 'soccer_england_league1', 'E3': 'soccer_england_league2',
    'SP1': 'soccer_spain_la_liga', 'SP2': 'soccer_spain_segunda_division',
    'D1': 'soccer_germany_bundesliga', 'D2': 'soccer_germany_bundesliga2',
    'F1': 'soccer_france_ligue_one', 'F2': 'soccer_france_ligue_two',
    'N1': 'soccer_netherlands_eredivisie', 'P1': 'soccer_portugal_primeira_liga', 
    'B1': 'soccer_belgium_pro_league', 'T1': 'soccer_turkey_super_league',
    'SC0': 'soccer_spfl_prem', 'G1': 'soccer_greece_super_league', 
    'A1': 'soccer_austria_bundesliga', 'SW1': 'soccer_switzerland_superleague',
    'ARG': 'soccer_argentina_primera', 'BRA': 'soccer_brazil_campeonato', 'CHN': 'soccer_china_superleague',
    'DNK': 'soccer_denmark_superliga', 'FIN': 'soccer_finland_veikkausliiga', 'IRL': 'soccer_ireland_premier_division',
    'JPN': 'soccer_japan_j_league', 'MEX': 'soccer_mexico_ligamx', 'NOR': 'soccer_norway_eliteserien',
    'POL': 'soccer_poland_ekstraklasa', 'SWE': 'soccer_sweden_allsvenskan', 'USA': 'soccer_usa_mls'
}

LEAGUE_COEFF = {
    'E0': 1.00, 'SP1': 0.96, 'I1': 0.94, 'D1': 0.92, 'F1': 0.88,
    'P1': 0.82, 'N1': 0.80, 'B1': 0.75, 'T1': 0.70, 'E1': 0.70, 'F2': 0.65,
    'SC0': 0.68, 'A1': 0.68, 'G1': 0.65, 'SW1': 0.65,
    'D2': 0.65, 'I2': 0.60, 'SP2': 0.60, 'E2': 0.55, 'E3': 0.50, 'EC': 0.45,
    'SC1': 0.50, 'SC2': 0.45, 'SC3': 0.40,
    'BRA': 0.75, 'ARG': 0.70, 'MEX': 0.65, 'USA': 0.65, 'JPN': 0.60
}

TEAM_MAPPING = {
    'Austria Wien': 'Austria Vienna', 'FC Blau-Weiß Linz': 'BW Linz', 'Grazer AK': 'GAK', 'Hartberg': 'Hartberg', 'LASK': 'LASK Linz', 'RB Salzburg': 'Salzburg', 'Red Bull Salzburg': 'Salzburg',
    'Rapid Wien': 'Rapid Vienna', 'Rheindorf Altach': 'Altach', 'Ried': 'Ried', 'Sturm Graz': 'Sturm Graz', 'SK Sturm Graz': 'Sturm Graz', 'WSG Tirol': 'Tirol', 'Wolfsberger AC': 'Wolfsberger',
    'Young Boys': 'Young Boys', 'Basel': 'Basel', '': 'Lausanne', 'Lugano': 'Lugano', 'Luzern': 'Luzern', 'Sion': 'Sion', 'FC St Gallen': 'St Gallen', 'Thun': 'Thun',
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
    'BSC Young Boys': 'Young Boys', 'FC Basel': 'Basel', 'FC Lausanne-Sport': 'Lausanne', 'FC Lugano': 'Lugano', 'Lugano': 'Lugano',
    'FC Luzern': 'Luzern', 'FC Sion': 'Sion', 'FC St Gallen': 'St Gallen', 'FC Thun': 'Thun', 'FC Winterthur': 'Winterthur', 'FC Zurich': 'Zurich', 'Grasshopper Zürich': 'Grasshoppers', 'Servette': 'Servette',
    'AE Kifisia FC': 'Kifisias', 'AEL': 'Larisa', 'Aris Thessaloniki': 'Aris', 'Atromitos Athens': 'Atromitos', 'Levadiakos': 'Levadiakos', 'PAOK Thessaloniki': 'PAOK', 'PAOK Salonika': 'PAOK',
    'Panetolikos Agrinio': 'Panetolikos', 'Panserraikos FC': 'Panserraikos', 'Volos FC': 'Volos NFC', 'Olympiakos Piraeus': 'Olympiakos', 'Panathinaikos FC': 'Panathinaikos', 'AEK Athens': 'AEK',
    'Basaksehir': 'Basaksehir', 'Istanbul Basaksehir': 'Basaksehir', 'Besiktas JK': 'Besiktas', 'Besiktas': 'Besiktas', 'Eyüpspor': 'Eyupspor', 'Fatih Karagümrük': 'Karagumruk',
    'Gazişehir Gaziantep': 'Gaziantep', 'Genclerbirligi SK': 'Genclerbirligi', 'Goztepe': 'Goztepe', 'Kasimpasa SK': 'Kasimpasa', 'Kasimpasa': 'Kasimpasa', 'Torku Konyaspor': 'Konyaspor', 'Çaykur Rizespor': 'Rizespor',
    'Galatasaray': 'Galatasaray', 'Fenerbahce': 'Fenerbahce', 'Trabzonspor': 'Trabzonspor', 'Celtic': 'Celtic', 'Rangers': 'Rangers', 'Rangers FC': 'Rangers',
    'Aberdeen': 'Aberdeen', 'Hearts': 'Hearts', 'KRC Genk': 'Genk', 'Union Saint-Gilloise': 'St Gilloise', 'AS Monaco': 'Monaco', 'AS Roma': 'Roma', 'Roma': 'Roma', 'Grimsby': 'Grimsby', 'Bodo/Glimt': 'Bodo/Glimt'
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
        open_1 = float(match_row.iloc[0]["Open_1"]); open_2 = float(match_row.iloc[0]["Open_2"])
        if open_1 > 0 and current_1 > 0 and ((open_1 - current_1) / open_1) * 100 >= 10.0:
            drop_alert += f"📉 DROP ALERT 1: Era {open_1:.2f} -> Ora {current_1:.2f} (-{((open_1 - current_1) / open_1) * 100:.1f}%)\n"
        if open_2 > 0 and current_2 > 0 and ((open_2 - current_2) / open_2) * 100 >= 10.0:
            drop_alert += f"📉 DROP ALERT 2: Era {open_2:.2f} -> Ora {current_2:.2f} (-{((open_2 - current_2) / open_2) * 100:.1f}%)\n"
        df_hist.loc[df_hist["MatchID"] == match_id, "Last_Update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_odds_history(df_hist)
    return drop_alert

def calculate_elo_updates(df_matches, league_code):
    base_rating = 1600 if LEAGUE_COEFF.get(league_code, 0.60) >= 0.90 else 1350 if LEAGUE_COEFF.get(league_code, 0.60) <= 0.55 else 1500 
    elo_dict = {}
    for t in set(df_matches['HomeTeam'].unique()) | set(df_matches['AwayTeam'].unique()): elo_dict[t] = base_rating
    for idx, row in df_matches.iterrows():
        h, a, res = row['HomeTeam'], row['AwayTeam'], row['Result']
        rh, ra = elo_dict[h], elo_dict[a]
        ea_home = 1 / (1 + 10 ** ((ra - rh) / 400)); ea_away = 1 / (1 + 10 ** ((rh - ra) / 400))
        sa_home, sa_away = (1.0, 0.0) if res == 'H' else (0.0, 1.0) if res == 'A' else (0.5, 0.5)
        elo_dict[h] = rh + 32 * (sa_home - ea_home); elo_dict[a] = ra + 32 * (sa_away - ea_away)
    return elo_dict

def calculate_synthetic_xg(row):
    try:
        hs, hst, hc = float(row.get('HS', 0)), float(row.get('HST', 0)), float(row.get('HC', 0))
        as_, ast, ac = float(row.get('AS', 0)), float(row.get('AST', 0)), float(row.get('AC', 0))
        return (hst * 0.32) + (max(0, hs - hst) * 0.05) + (hc * 0.03), (ast * 0.32) + (max(0, as_ - ast) * 0.05) + (ac * 0.03)
    except: return 0.0, 0.0

@st.cache_data(ttl=3600)
def scarica_dati(codice_lega):
    if codice_lega in ['UCL', 'UEL', 'UECL']: return None, None, None, None
    try: df = pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{STAGIONE}/{codice_lega}.csv")
    except:
        try: df = pd.read_csv(f"https://www.football-data.co.uk/new/{codice_lega}.csv")
        except: return None, None, None, None

    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date')
        df['HomeTeam'] = df['HomeTeam'].str.strip(); df['AwayTeam'] = df['AwayTeam'].str.strip()
        if not all(col in df.columns for col in ['Date','HomeTeam','AwayTeam','FTHG','FTAG']): return None, None, None, None
        
        for col in ['HST','AST','HC','AC','HF','AF','HY','AY', 'HTHG', 'HTAG']:
            if col not in df.columns: df[col] = 0.0
            else: df[col] = df[col].fillna(0.0)
        
        df['xG_H'], df['xG_A'] = zip(*df.apply(calculate_synthetic_xg, axis=1))
        df['Result'] = np.where(df['FTHG'] > df['FTAG'], 'H', np.where(df['FTHG'] < df['FTAG'], 'A', 'D'))
        
        elo_ratings = calculate_elo_updates(df, codice_lega)

        avgs = {
            'Goals_H': df['FTHG'].mean(), 'Goals_A': df['FTAG'].mean(), 'xG_H': df['xG_H'].mean(), 'xG_A': df['xG_A'].mean(),
            'Shots_H': df['HST'].mean(), 'Shots_A': df['AST'].mean(), 'Corn_H': df['HC'].mean(), 'Corn_A': df['AC'].mean(),
            'Fouls_H': df['HF'].mean(), 'Fouls_A': df['AF'].mean(), 'Cards_H': df['HY'].mean(), 'Cards_A': df['AY'].mean(),
            'HT_H': df['HTHG'].mean(), 'HT_A': df['HTAG'].mean(),
        }
        
        h_df = df[['Date','HomeTeam','Result']].rename(columns={'HomeTeam':'Team'})
        h_df['IsHome'] = 1; h_df['FormChar'] = np.where(h_df['Result'] == 'H', 'W', np.where(h_df['Result'] == 'A', 'L', 'D'))
        h_df['Goals_For'] = df['FTHG']; h_df['Goals_Ag'] = df['FTAG']; h_df['xG_For'] = df['xG_H']; h_df['xG_Ag'] = df['xG_A'] 
        h_df['Shots_For'] = df['HST']; h_df['Shots_Ag'] = df['AST']; h_df['Corn_For'] = df['HC']; h_df['Corn_Ag'] = df['AC']
        h_df['Fouls_For'] = df['HF']; h_df['Fouls_Ag'] = df['AF']; h_df['Cards_For'] = df['HY']; h_df['Cards_Ag'] = df['AY']
        h_df['HT_For'] = df['HTHG']; h_df['HT_Ag'] = df['HTAG']
        
        a_df = df[['Date','AwayTeam','Result']].rename(columns={'AwayTeam':'Team'})
        a_df['IsHome'] = 0; a_df['FormChar'] = np.where(a_df['Result'] == 'A', 'W', np.where(a_df['Result'] == 'H', 'L', 'D'))
        a_df['Goals_For'] = df['FTAG']; a_df['Goals_Ag'] = df['FTHG']; a_df['xG_For'] = df['xG_A']; a_df['xG_Ag'] = df['xG_H']
        a_df['Shots_For'] = df['AST']; a_df['Shots_Ag'] = df['HST']; a_df['Corn_For'] = df['AC']; a_df['Corn_Ag'] = df['HC']
        a_df['Fouls_For'] = df['AF']; a_df['Fouls_Ag'] = df['HF']; a_df['Cards_For'] = df['AY']; a_df['Cards_Ag'] = df['HY']
        a_df['HT_For'] = df['HTAG']; a_df['HT_Ag'] = df['HTHG']
        
        full_df = pd.concat([h_df, a_df]).sort_values(['Team','Date'])
        
        for m in ['Goals', 'xG', 'Shots', 'Corn', 'Fouls', 'Cards', 'HT']:
            full_df[f'{m}_Att_Rat'] = np.where(full_df['IsHome']==1, np.where(avgs[f'{m}_H']>0, full_df[f'{m}_For']/avgs[f'{m}_H'], 1.0), np.where(avgs[f'{m}_A']>0, full_df[f'{m}_For']/avgs[f'{m}_A'], 1.0))
            full_df[f'{m}_Def_Rat'] = np.where(full_df['IsHome']==1, np.where(avgs[f'{m}_A']>0, full_df[f'{m}_Ag']/avgs[f'{m}_A'], 1.0), np.where(avgs[f'{m}_H']>0, full_df[f'{m}_Ag']/avgs[f'{m}_H'], 1.0))
            full_df[f'W_{m}_Att'] = full_df.groupby('Team')[f'{m}_Att_Rat'].transform(lambda x: x.ewm(span=5, min_periods=1).mean())
            full_df[f'W_{m}_Def'] = full_df.groupby('Team')[f'{m}_Def_Rat'].transform(lambda x: x.ewm(span=5, min_periods=1).mean())

        return full_df, df, avgs, elo_ratings
    except: return None, None, None, None

def get_live_matches(api_key, sport_key):
    if not sport_key: return []
    try: return requests.get(f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions={REGION}&markets={MARKET}').json()
    except: return []

def calculate_kelly_stake(bankroll, odds, probability, fraction=0.3):
    if odds <= 1 or probability <= 0: return 0.0
    b = odds - 1; q = 1 - probability
    return round(bankroll * max(0, (b * probability - q) / b) * fraction, 2)

def get_dc_matrix(lam_h, lam_a):
    mat = np.zeros((10, 10))
    for i in range(10):
        for j in range(10): mat[i,j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
    rho = 0.13
    mat[0,0] *= (1 - (lam_h * lam_a * rho)); mat[0,1] *= (1 + (lam_h * rho))
    mat[1,0] *= (1 + (lam_a * rho)); mat[1,1] *= (1 - rho)
    return mat / np.sum(mat)

def calcola_1x2_dixon_coles(lam_h, lam_a):
    mat = get_dc_matrix(lam_h, lam_a)
    p1 = np.sum(np.tril(mat,-1)); pX = np.trace(mat); p2 = np.sum(np.triu(mat,1))
    return (1/p1 if p1>0 else 99), (1/pX if pX>0 else 99), (1/p2 if p2>0 else 99)

def calcola_h2h_1x2(val_h, val_a):
    joint = np.outer(poisson.pmf(np.arange(40), val_h), poisson.pmf(np.arange(40), val_a))
    return np.sum(np.tril(joint, -1)), np.trace(joint), np.sum(np.triu(joint, 1))

def find_team_stats_global(team_name, cache_dataframes):
    for league_code, (df_weighted, _, averages, elo_dict) in cache_dataframes.items():
        if df_weighted is None: continue
        team_stats = df_weighted[df_weighted['Team'] == team_name]
        if not team_stats.empty:
            return team_stats.iloc[-1], elo_dict.get(team_name, 1500), averages, "-".join(df_weighted[df_weighted['Team'] == team_name].tail(5)['FormChar'].tolist()), league_code
    return None, 1500, None, "N/A", "N/A"

# ==============================================================================
# FUNZIONI AI (GOOGLE GEMINI) - FIXED 404
# ==============================================================================

def genera_analisi_risk_management(gemini_api_key, h_team, a_team, exp_data, roi_1x2, mq1, mqx, mq2, preview_text):
    try:
        genai.configure(api_key=gemini_api_key)
        
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        safe_models = [m for m in available_models if '2.0' not in m and 'vision' not in m]
        
        if not safe_models: return "❌ Errore API: Modelli non disponibili."
            
        model_name = safe_models[0]
        for m in safe_models:
            if '1.5-flash' in m:
                model_name = m; break
                
        model = genai.GenerativeModel(model_name=model_name)
        
        prompt = f"""Agisci come un Risk Manager professionista di betting sportivo.
Analizza il match {h_team} vs {a_team}.
I nostri modelli matematici hanno prodotto questi numeri:
- Quote Reali Calcolate (1X2): 1 ({mq1:.2f}), X ({mqx:.2f}), 2 ({mq2:.2f})
- Vantaggio Matematico (ROI): 1 ({roi_1x2['1']*100:.1f}%), X ({roi_1x2['X']*100:.1f}%), 2 ({roi_1x2['2']*100:.1f}%)

TESTO DELL'ANTEPRIMA FORNITO DALL'ANALISTA (Leggilo attentamente):
"{preview_text}"

Esegui questa procedura rigorosa basandoti ESCLUSIVAMENTE sul testo fornito:
1. Identifica le motivazioni chiave (es. lotta retrocessione, coppe europee).
2. Estrai i giocatori infortunati o squalificati confermati nel testo.
3. Ignora le raccomandazioni di scommessa dell'autore del testo (es. "ci aspettiamo entrambe le squadre a segno"). Devi decidere TU incrociando il testo con i NOSTRI dati ROI.

Restituisci l'output usando ESATTAMENTE questo formato:

- IL FATTO: (Sintesi secca di motivazioni, infortuni e squalifiche lette nel testo. Es. "Tottenham in lotta retrocessione, squalificato Romero. Palace privo di Mateta e Lerma"). Se non hai incollato alcun testo, scrivi "Nessun dato fornito".
- IL SEGNALE: (Come questi fatti impattano il nostro ROI. Es. "Le assenze difensive del Palace e le motivazioni di sopravvivenza del Tottenham rafforzano/indeboliscono la quota...").
- IL CONTRO-CANTO: (Qual è il bias narrativo del mercato e qual è la mossa giusta per noi. Azione: Conferma Bet, No Bet, Switch Mercato).

Tono oggettivo, sintetico, privo di moralismi."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Errore API: {str(e)}"

# ==============================================================================
# MODIFICA ALL'INTERFACCIA (DA INSERIRE DOVE C'È IL PULSANTE AI)
# ==============================================================================
# Sostituisci la parte di rendering delle tab (alla fine del file) con questa:

    # RENDERIZZAZIONE TAB PRINCIPALE
    if st.session_state['results_data']:
        active_leagues = [l for l in st.session_state['results_data'] if st.session_state['results_data'][l]]
        if active_leagues:
            tabs = st.tabs(active_leagues)
            for i, l in enumerate(active_leagues):
                with tabs[i]:
                    for m in st.session_state['results_data'][l]:
                        with st.expander(m['label']):
                            st.markdown(m['html'], unsafe_allow_html=True)
                            
                            # AI BUTTON CON TEXT AREA (Tab Principale)
                            if m['is_top_5']:
                                display_key = f"ai_res_{m['match_id']}"
                                text_input_key = f"txt_{m['match_id']}"
                                
                                st.markdown("---")
                                st.markdown("📝 **Iniezione Dati Qualitativi (Opzionale)**")
                                preview_text = st.text_area("Incolla qui l'anteprima da Diretta.it o Sofascore:", key=text_input_key, height=100)
                                
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    if st.button("🧠 Genera Risk Management", key=f"btn_{m['match_id']}_main"):
                                        if not gemini_key_input:
                                            st.error("Inserisci la chiave Gemini a sinistra!")
                                        elif not preview_text.strip():
                                            st.warning("⚠️ Incolla un testo di anteprima prima di generare l'analisi!")
                                        else:
                                            with st.spinner("Analisi Risk Management in corso..."):
                                                d = m['ai_data']
                                                res = genera_analisi_risk_management(gemini_key_input, d['h'], d['a'], d['exp'], d['roi'], d['q1'], d['qx'], d['q2'], preview_text)
                                                st.session_state[display_key] = res
                                
                                if display_key in st.session_state:
                                    st.markdown(f"<div class='ai-box'>{st.session_state[display_key]}</div>", unsafe_allow_html=True)
# ==============================================================================
# GENERATORE UI TERMINALE
# ==============================================================================

def generate_complete_terminal(h_team, a_team, exp_data, odds_1x2, roi_1x2, min_prob, last_date_h, last_date_a, bankroll, h_form, a_form, league_code, h_elo, a_elo, fatigue_alert, drop_alert):
    html = f"""<div class='terminal-box'>"""
    max_date = max(last_date_h, last_date_a); days_lag = (datetime.now() - max_date).days
    
    html += f"<div class='term-header'>[ELO] {h_team} ({int(h_elo)}) vs {a_team} ({int(a_elo)})</div>"
    if h_elo - a_elo > 100: html += f"<span class='term-green'>>>> ELO FAVORITE: {h_team} (+{int(h_elo - a_elo)})</span>\n"
    elif a_elo - h_elo > 100: html += f"<span class='term-green'>>>> ELO FAVORITE: {a_team} (+{int(abs(h_elo - a_elo))})</span>\n"
    
    if fatigue_alert: html += f"<div class='term-fatigue'>{fatigue_alert}</div>\n"
    if drop_alert: html += f"<div class='term-drop'>{drop_alert}</div>\n"
    if days_lag > 14: html += f"<div class='term-warn'>⚠️ DATI VECCHI ({days_lag}gg)</div>\n"
    html += f"FORMA: {h_team:<15} [{h_form}] vs [{a_form}] {a_team}\n"
    
    html += f"\n<span class='term-section'>[ 1X2 & VALUE BET ]</span>\n"
    html += f"{'SEGNO':<6} | {'MY QUOTA':<10} | {'BOOKIE':<8} | {'VALUE':<8} | {'STAKE'}\n" + "-"*60 + "\n"
    
    for segno, roi, book_q in [('1', roi_1x2['1'], odds_1x2['1']), ('X', roi_1x2['X'], odds_1x2['X']), ('2', roi_1x2['2'], odds_1x2['2'])]:
        my_q = book_q / (roi + 1) if (roi+1) > 0 else 99.0; prob = 1/my_q if my_q > 0 else 0
        stake = calculate_kelly_stake(bankroll, book_q, prob) if roi > 0 else 0.0
        val_str = f"<span class='term-val'>{roi*100:+.0f}% (TOP)</span>" if roi >= 0.15 and book_q <= 4.0 and prob >= min_prob else f"<span class='term-green'>{roi*100:+.0f}%</span>" if roi > 0 and prob >= min_prob else f"<span class='term-dim'>{roi*100:+.0f}%</span>"
        html += f"{segno:<6} | {my_q:<10.2f} | {book_q:<8.2f} | {val_str:<20} | {f'€ {stake:.2f}' if stake > 0 else '-'}\n"

    html += f"\n<span class='term-section'>[ ENGINE METRICS ]</span>\n"
    html += f"CASA → Real Goals Exp: {exp_data['RealGoals'][0]:.2f} | xG Exp: {exp_data['xG'][0]:.2f}\n"
    html += f"OSP  → Real Goals Exp: {exp_data['RealGoals'][1]:.2f} | xG Exp: {exp_data['xG'][1]:.2f}\n"

    html += f"\n<span class='term-section'>[ TESTA A TESTA ]</span>\n"
    for label, key in [("GOL", 'Goals'), ("CORNER", 'Corn'), ("TIRI", 'Shots'), ("FALLI", 'Fouls'), ("CARDS", 'Cards')]:
        vh, va = exp_data[key]; p1, pX, p2 = calcola_h2h_1x2(vh, va)
        s1, sx, s2 = f"{p1*100:04.1f}%", f"{pX*100:04.1f}%", f"{p2*100:04.1f}%"
        if p1 >= min_prob: s1 = f"<span class='term-green'>{s1}</span>"
        if pX >= min_prob: sx = f"<span class='term-green'>{sx}</span>"
        if p2 >= min_prob: s2 = f"<span class='term-green'>{s2}</span>"
        html += f"{label:<10}: 1 ({s1}) | X ({sx}) | 2 ({s2})   [Exp: {vh:.1f} vs {va:.1f}]\n"

    mat = get_dc_matrix(exp_data['Goals'][0], exp_data['Goals'][1])
    prob_ng = mat[0, :].sum() + mat[:, 0].sum() - mat[0,0]; prob_gol = 1 - prob_ng
    
    html += "\n<span class='term-section'>[ MERCATI SPECIALI GOL ]</span>\n"
    html += f"GOL (Entrambe Segnano) : {prob_gol*100:04.1f}% | Quota: {1/prob_gol if prob_gol>0 else 99:.2f}\n"
    html += f"NO GOL                 : {prob_ng*100:04.1f}% | Quota: {1/prob_ng if prob_ng>0 else 99:.2f}\n"

    html += "\n<span class='term-section'>[ TOP 3 RISULTATI ESATTI ]</span>\n"
    flat_mat = mat.flatten(); top_3_idx = flat_mat.argsort()[-3:][::-1]
    for idx in top_3_idx:
        p = flat_mat[idx]; html += f"{idx // 10}-{idx % 10} ({p*100:04.1f}% | Q: {1/p if p>0 else 99:.2f})\n"

    ht_lam = exp_data['HT'][0] + exp_data['HT'][1]
    html += f"\n<span class='term-section'>[ ANALISI PRIMO TEMPO (HT) ]</span>\n"
    html += f"CASA → Real Goals HT: {exp_data['HT'][0]:.2f}\nOSP  → Real Goals HT: {exp_data['HT'][1]:.2f}\n\n"
    html += f"Over 0.5 Primo Tempo : {(1 - poisson.pmf(0, ht_lam))*100:04.1f}% | Quota: {1/(1 - poisson.pmf(0, ht_lam)) if (1 - poisson.pmf(0, ht_lam))>0 else 99:.2f}\n"
    html += f"Over 1.5 Primo Tempo : {(1 - poisson.pmf(0, ht_lam) - poisson.pmf(1, ht_lam))*100:04.1f}% | Quota: {1/(1 - poisson.pmf(0, ht_lam) - poisson.pmf(1, ht_lam)) if (1 - poisson.pmf(0, ht_lam) - poisson.pmf(1, ht_lam))>0 else 99:.2f}\n"

    prop_configs = [("GOL", exp_data['Goals'], [0.5, 1.5, 2.5], [0.5, 1.5], [1.5, 2.5, 3.5])]
    if league_code not in COMPACT_LEAGUES: prop_configs.extend([("CORNER", exp_data['Corn'], [3.5, 4.5, 5.5], [2.5, 3.5, 4.5], [8.5, 9.5, 10.5])])
    
    for label, (eh, ea), r_h, r_a, r_tot in prop_configs:
        html += f"\n<span class='term-section'>[ {label} DETTAGLIO ]</span>\n{'LINEA':<15} | {'PROB %':<8} | {'QUOTA'}\n" + "-"*40 + "\n"
        def add_rows(prefix, r, exp):
            res = ""
            for l in r:
                p = poisson.sf(int(l), exp); q = 1/p if p > 0 else 99
                row_str = f"{prefix+' Ov '+str(l):<15} | {p*100:04.1f}%   | {q:.2f}"
                res += f"<span class='term-green'>{row_str}</span>\n" if p >= min_prob else f"<span class='term-dim'>{row_str}</span>\n"
            return res
        html += add_rows("CASA", r_h, eh) + add_rows("OSP", r_a, ea) + add_rows("TOT", r_tot, eh+ea)

    html += "</div>"
    return html

# ==============================================================================
# INTERFACCIA PRINCIPALE
# ==============================================================================

with st.sidebar:
    st.header("🎛️ Configurazione")
    api_key_input = st.text_input("The Odds API Key", type="password")
    gemini_key_input = st.text_input("Gemini API Key (Opzionale per AI)", type="password")
    bankroll_input = st.number_input("Bankroll (€)", min_value=10.0, value=26.50, step=0.5)
    
    st.divider()
    min_prob_val = st.slider("Probabilità Minima (Verde)", 0.50, 0.90, 0.65, step=0.05)
    
    st.divider()
    st.markdown("### 🏆 Campionati")
    
    active_groups = []
    col1, col2 = st.columns(2)
    for idx, (g_name, g_codes) in enumerate(LEAGUE_GROUPS.items()):
        with col1 if idx % 2 == 0 else col2:
            if st.checkbox(g_name, value=(g_name == "🏆 Top 5 (Tier 1)")):
                active_groups.extend(g_codes)
                
    st.markdown("#### 2️⃣ Selezione Manuale")
    manual_selection = st.multiselect("Aggiungi Leghe:", options=sorted(list(ALL_LEAGUES.keys())), format_func=lambda x: f"{ALL_LEAGUES[x]} ({x})", default=[])
    final_selection_codes = list(set(active_groups + manual_selection))
    st.caption(f"Totale leghe selezionate: {len(final_selection_codes)}")
    
    st.divider()
    inspect_csv_mode = st.checkbox("🔍 ISPEZIONA CSV", value=False)

st.title("SmartBet Pro 63.1")
st.caption("Engine: Risk Management AI | Deep Data | Exact Score | HT Engine | Dropping Odds")

tab_main, tab_cal = st.tabs(["🚀 ANALISI MATCH", "📅 CALENDARIO"])

with tab_main:
    start_analisys = st.button("🚀 CERCA VALUE BETS", type="primary", use_container_width=True)

    if start_analisys:
        if not api_key_input: st.error("Inserisci API Key di The Odds API!")
        elif not final_selection_codes: st.error("Seleziona almeno una lega!")
        else:
            st.session_state['results_data'] = {}; st.session_state['calendar_data'] = []
            domestic_cache = {}
            leagues_to_load = [k for k in ALL_LEAGUES.keys() if k not in ['UCL','UEL','UECL']] if any(c in ['UCL','UEL','UECL'] for c in final_selection_codes) else [k for k in final_selection_codes if k not in ['UCL','UEL','UECL']]
            
            status = st.empty(); status.text("Caricamento database statistici & Calcolo ELO...")
            for code in leagues_to_load: domestic_cache[code] = scarica_dati(code)
                
            progress = st.progress(0); total_steps = len(final_selection_codes)
            
            for idx, code in enumerate(final_selection_codes):
                progress.progress((idx+1)/total_steps)
                league_name = ALL_LEAGUES.get(code, code); status.text(f"Analisi: {league_name}...")
                if league_name not in st.session_state['results_data']: st.session_state['results_data'][league_name] = []
                
                matches = get_live_matches(api_key_input, API_MAPPING.get(code, ''))
                if matches:
                    for m in matches:
                        if 'home_team' not in m: continue
                        h_raw, a_raw = m['home_team'], m['away_team']
                        h_team = TEAM_MAPPING.get(h_raw, h_raw); a_team = TEAM_MAPPING.get(a_raw, a_raw)
                        raw_date_obj, fmt_date_str = parse_date(m.get('commence_time', ''))
                        
                        h_data, h_elo, h_avgs, h_form, h_lg = find_team_stats_global(h_team, domestic_cache)
                        a_data, a_elo, a_avgs, a_form, a_lg = find_team_stats_global(a_team, domestic_cache)
                        
                        if h_data is None or a_data is None: continue
                        
                        q1_b, qX_b, q2_b = 0, 0, 0
                        if m.get('bookmakers'):
                            odds_1, odds_X, odds_2 = [], [], []
                            for b in m['bookmakers']:
                                for mk in b['markets']:
                                    if mk['key'] == 'h2h':
                                        for o in mk['outcomes']:
                                            if o['name'] == h_raw: odds_1.append(o['price'])
                                            elif o['name'] == 'Draw': odds_X.append(o['price'])
                                            elif o['name'] == a_raw: odds_2.append(o['price'])
                            q1_b = max(odds_1) if odds_1 else 0; qX_b = max(odds_X) if odds_X else 0; q2_b = max(odds_2) if odds_2 else 0

                        drop_alert = check_dropping_odds(h_team, a_team, str(raw_date_obj), q1_b, qX_b, q2_b)

                        exp_data = {} 
                        for met in ['Goals', 'xG', 'Shots', 'Corn', 'Fouls', 'Cards', 'HT']:
                            val_h = h_data[f'W_{met}_Att'] * a_data[f'W_{met}_Def'] * h_avgs[f'{met}_H']
                            val_a = a_data[f'W_{met}_Att'] * h_data[f'W_{met}_Def'] * a_avgs[f'{met}_A']
                            exp_data[met] = (val_h, val_a)
                        exp_data['RealGoals'] = exp_data['Goals'] 

                        mix_h, mix_a = (exp_data['Goals'][0] * 0.5 + exp_data['xG'][0] * 0.5, exp_data['Goals'][1] * 0.5 + exp_data['xG'][1] * 0.5) if exp_data['xG'][0] > 0 and exp_data['xG'][1] > 0 else exp_data['Goals']
                        if h_elo - a_elo > 100: mix_h *= 1.05
                        elif a_elo - h_elo > 100: mix_a *= 1.05

                        fatigue_alert = ""
                        if (raw_date_obj - h_data['Date'].date()).days < 4: mix_h *= 0.85; fatigue_alert += f"⚠️ STANCHEZZA: {h_team}\n"
                        if (raw_date_obj - a_data['Date'].date()).days < 4: mix_a *= 0.85; fatigue_alert += f"⚠️ STANCHEZZA: {a_team}\n"
                        
                        exp_data['Goals'] = (mix_h, mix_a)
                        if q1_b == 0: continue
                        
                        my_q1, my_qX, my_q2 = calcola_1x2_dixon_coles(mix_h, mix_a)
                        roi_1x2 = {'1': ((1/my_q1)*q1_b)-1, 'X': ((1/my_qX)*qX_b)-1, '2': ((1/my_q2)*q2_b)-1}
                        
                        html_block = generate_complete_terminal(h_team, a_team, exp_data, {'1':q1_b,'X':qX_b,'2':q2_b}, roi_1x2, min_prob_val, h_data['Date'], a_data['Date'], bankroll_input, h_form, a_form, code, h_elo, a_elo, fatigue_alert, drop_alert)
                        
                        # Struttura Dati per sincronizzazione UI
                        match_id = f"{h_team}_{a_team}_{raw_date_obj}"
                        item_ok = {
                            'match_id': match_id,
                            'label': f"✅ {fmt_date_str} | {h_team} vs {a_team} ({code})", 
                            'html': html_block, 'raw_date': raw_date_obj,
                            'is_top_5': code in TOP_5_LEAGUES,
                            'ai_data': {'h': h_team, 'a': a_team, 'exp': exp_data, 'roi': roi_1x2, 'q1': my_q1, 'qx': my_qX, 'q2': my_q2}
                        }
                        st.session_state['results_data'][league_name].append(item_ok)
                        st.session_state['calendar_data'].append(item_ok)

            status.empty(); st.success("Analisi Completata.")

    # RENDERIZZAZIONE TAB PRINCIPALE
    if st.session_state['results_data']:
        active_leagues = [l for l in st.session_state['results_data'] if st.session_state['results_data'][l]]
        if active_leagues:
            tabs = st.tabs(active_leagues)
            for i, l in enumerate(active_leagues):
                with tabs[i]:
                    for m in st.session_state['results_data'][l]:
                        with st.expander(m['label']):
                            st.markdown(m['html'], unsafe_allow_html=True)
                            
                            # AI BUTTON (Tab Principale)
                            if m['is_top_5']:
                                display_key = f"ai_res_{m['match_id']}"
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    if st.button("🧠 Risk Management AI", key=f"btn_{m['match_id']}_main"):
                                        if not gemini_key_input:
                                            st.error("Inserisci la chiave Gemini a sinistra!")
                                        else:
                                            with st.spinner("Ricerca sul web in corso..."):
                                                d = m['ai_data']
                                                res = genera_analisi_risk_management(gemini_key_input, d['h'], d['a'], d['exp'], d['roi'], d['q1'], d['qx'], d['q2'])
                                                st.session_state[display_key] = res
                                
                                if display_key in st.session_state:
                                    st.markdown(f"<div class='ai-box'>{st.session_state[display_key]}</div>", unsafe_allow_html=True)

# RENDERIZZAZIONE TAB CALENDARIO (Ora Sincronizzata)
with tab_cal:
    st.markdown("### 📅 Calendario")
    if st.session_state['calendar_data']:
        unique_dates = sorted(list(set([x['raw_date'] for x in st.session_state['calendar_data']])))
        if unique_dates:
            selected_date = st.date_input("Seleziona Data:", value=unique_dates[0], min_value=unique_dates[0])
            daily_matches = [x for x in st.session_state['calendar_data'] if x['raw_date'] == selected_date]
            if daily_matches:
                for m in daily_matches:
                    with st.expander(m['label']): 
                        st.markdown(m['html'], unsafe_allow_html=True)
                        
                        # AI BUTTON (Tab Calendario - Sync)
                        if m.get('is_top_5'):
                            display_key = f"ai_res_{m['match_id']}"
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                if st.button("🧠 Risk Management AI", key=f"btn_{m['match_id']}_cal"):
                                    if not gemini_key_input:
                                        st.error("Inserisci la chiave Gemini a sinistra!")
                                    else:
                                        with st.spinner("Ricerca sul web in corso..."):
                                            d = m['ai_data']
                                            res = genera_analisi_risk_management(gemini_key_input, d['h'], d['a'], d['exp'], d['roi'], d['q1'], d['qx'], d['q2'])
                                            st.session_state[display_key] = res
                            
                            if display_key in st.session_state:
                                st.markdown(f"<div class='ai-box'>{st.session_state[display_key]}</div>", unsafe_allow_html=True)
            else: st.warning("Nessuna partita in questa data.")
    else: st.info("Esegui prima la ricerca dei Value Bets per popolare il calendario.")
