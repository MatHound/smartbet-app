import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from datetime import datetime, timedelta

# ==============================================================================
# 1. CONFIGURAZIONE E COSTANTI
# ==============================================================================
st.set_page_config(page_title="SmartBet Real Form", page_icon="📈", layout="wide")

# COSTANTI GLOBALI
STAGIONE = "2526"
REGION = 'eu'
MARKET = 'h2h'

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
    .streamlit-expanderHeader { font-weight: bold; background-color: #f0f2f6; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- DATABASE LEGHE ---
LEAGUE_GROUPS = {
    "🇪🇺 Coppe Europee": ['UCL', 'UEL', 'UECL'],
    "🏆 Top 5 (Tier 1)": ['I1', 'E0', 'SP1', 'D1', 'F1'],
    "⚽ Europe Tier 2": ['N1', 'P1', 'B1', 'T1', 'SC0', 'G1', 'A1', 'SW1'],
    "📉 Leghe Minori": ['I2', 'E1', 'E2', 'D2', 'SP2']
}

ALL_LEAGUES = {
    'UCL': 'Champions League', 'UEL': 'Europa League', 'UECL': 'Conference League',
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

# --- MEGA MAPPING 44.0 ---
TEAM_MAPPING = {
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
    'Südtirol': 'Sudtirol', 'US Catanzaro 1929': 'Catanzaro',
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
# 2. FUNZIONI CORE
# ==============================================================================

def parse_date(iso_date_str):
    try:
        dt = datetime.strptime(iso_date_str, "%Y-%m-%dT%H:%M:%SZ")
        dt_ita = dt + timedelta(hours=1) 
        return dt_ita.date(), dt_ita.strftime("%d/%m %H:%M")
    except:
        return datetime.now().date(), "Oggi"

@st.cache_data(ttl=3600)
def scarica_dati(codice_lega):
    if codice_lega in ['UCL', 'UEL', 'UECL']: return None, None
    url = f"https://www.football-data.co.uk/mmz4281/{STAGIONE}/{codice_lega}.csv"
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date')
        df['HomeTeam'] = df['HomeTeam'].str.strip(); df['AwayTeam'] = df['AwayTeam'].str.strip()
        
        needed = ['Date','HomeTeam','AwayTeam','FTHG','FTAG']
        if not all(col in df.columns for col in needed): return None, None
        
        for col in ['HST','AST','HC','AC','HF','AF','HY','AY']:
            if col not in df.columns: df[col] = 0.0
            
        # --- NEW ENGINE: UNIFIED FORM (HOME + AWAY) ---
        # Creiamo un dataset "Lung" (Verticale) dove ogni riga è una performance
        # Rinominiamo le colonne per avere nomi standard: Team, Goals, Shots, Corn, Fouls, Cards
        
        h_stats = df[['Date','HomeTeam','FTHG','HST','HC','HF','HY']].rename(
            columns={'HomeTeam':'Team', 'FTHG':'Goals', 'HST':'Shots', 'HC':'Corn', 'HF':'Fouls', 'HY':'Cards'}
        )
        a_stats = df[['Date','AwayTeam','FTAG','AST','AC','AF','AY']].rename(
            columns={'AwayTeam':'Team', 'FTAG':'Goals', 'AST':'Shots', 'AC':'Corn', 'AF':'Fouls', 'AY':'Cards'}
        )
        
        # Uniamo tutto e ordiniamo per data
        full_df = pd.concat([h_stats, a_stats]).sort_values(['Team','Date'])
        
        # Calcoliamo la media pesata (span=5) su TUTTE le partite (Casa + Fuori)
        # Questo cattura la "Forma Reale"
        cols = ['Goals','Shots','Corn','Fouls','Cards']
        for c in cols:
            full_df[f'W_{c}'] = full_df.groupby('Team')[c].transform(lambda x: x.ewm(span=5).mean())
            
        # Per compatibilità, ritorniamo anche df originale
        return full_df, df 
    except: return None, None

def get_live_matches(api_key, sport_key):
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions={REGION}&markets={MARKET}'
    try: return requests.get(url).json()
    except: return []

def calcola_1x2_lambda(exp_goals_h, exp_goals_a):
    # I dati in input SONO GIA' GOL ATTESI REALI.
    # NESSUN MOLTIPLICATORE AGGIUNTIVO!
    
    # Minimo floor per evitare 0 assoluti (che rompono Poisson)
    lam_h = exp_goals_h if exp_goals_h > 0.1 else 0.1
    lam_a = exp_goals_a if exp_goals_a > 0.1 else 0.1
    
    mat = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            mat[i,j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            
    # Dixon-Coles Correction
    rho = 0.13
    mat[0,0] *= (1 - (lam_h * lam_a * rho))
    mat[0,1] *= (1 + (lam_h * rho))
    mat[1,0] *= (1 + (lam_a * rho))
    mat[1,1] *= (1 - rho)
    
    # Normalizza perché DC altera la somma totale
    mat = mat / np.sum(mat)
    
    p1 = np.sum(np.tril(mat,-1)); pX = np.trace(mat); p2 = np.sum(np.triu(mat,1))
    return (1/p1 if p1>0 else 99), (1/pX if pX>0 else 99), (1/p2 if p2>0 else 99), lam_h, lam_a

def calcola_h2h_favorito(val_h, val_a):
    r = np.arange(40)
    pmf_h = poisson.pmf(r, val_h); pmf_a = poisson.pmf(r, val_a)
    joint = np.outer(pmf_h, pmf_a)
    p_h = np.sum(np.tril(joint, -1)); p_a = np.sum(np.triu(joint, 1))
    return p_h, p_a

def find_team_stats_global(team_name, cache_dataframes):
    for league_code, (df_weighted, _) in cache_dataframes.items():
        if df_weighted is None: continue
        team_stats = df_weighted[df_weighted['Team'] == team_name]
        if not team_stats.empty:
            last_row = team_stats.iloc[-1]
            coeff = LEAGUE_COEFF.get(league_code, 0.65)
            return last_row, coeff, league_code
    return None, 0, None

def generate_missing_data_terminal(h_team, a_team, h_found, a_found, bookie_odds):
    html = f"""<div class='terminal-missing'>"""
    html += f"<span style='color:#FF5555; font-weight:bold;'>[ ! ] DATI INSUFFICIENTI: {h_team} vs {a_team}</span>\n"
    html += "-"*60 + "\n"
    if not h_found: html += f"❌ Dati Storici mancanti per: {h_team}\n"
    else: html += f"✅ Dati Storici OK per: {h_team}\n"
    if not a_found: html += f"❌ Dati Storici mancanti per: {a_team}\n"
    else: html += f"✅ Dati Storici OK per: {a_team}\n"
    html += "\nINFO BOOKMAKER (Solo per riferimento):\n"
    html += f"1: {bookie_odds['1']:.2f} | X: {bookie_odds['X']:.2f} | 2: {bookie_odds['2']:.2f}\n"
    html += "</div>"
    return html

def generate_complete_terminal(h_team, a_team, stats, lam_h, lam_a, odds_1x2, roi_1x2, min_prob, last_date_h, last_date_a):
    html = f"""<div class='terminal-box'>"""
    
    max_date = max(last_date_h, last_date_a)
    days_lag = (datetime.now() - max_date).days
    if days_lag > 14: html += f"<div class='term-warn'>⚠️ DATI DATATI ({days_lag}gg). ATTENZIONE.</div>\n"
    
    html += f"<span class='term-section'>[ 1X2 ANALYSIS ]</span>\n"
    html += f"{'SEGNO':<6} | {'MY QUOTA':<10} | {'BOOKIE':<8} | {'VALUE'}\n"
    html += "-"*45 + "\n"
    segni = [('1', roi_1x2['1'], odds_1x2['1']), ('X', roi_1x2['X'], odds_1x2['X']), ('2', roi_1x2['2'], odds_1x2['2'])]
    for segno, roi, book_q in segni:
        my_q = book_q / (roi + 1) if (roi+1) > 0 else 99.0
        val_str = f"{roi*100:+.0f}%"
        implied_prob = 1/my_q if my_q > 0 else 0
        if roi >= 0.15 and book_q <= 4.0 and implied_prob >= min_prob: val_str = f"<span class='term-val'>{val_str} (TOP)</span>"
        elif roi > 0 and implied_prob >= min_prob: val_str = f"<span class='term-green'>{val_str}</span>"
        else: val_str = f"<span class='term-dim'>{val_str}</span>"
        html += f"{segno:<6} | {my_q:<10.2f} | {book_q:<8.2f} | {val_str}\n"

    html += f"\n<span class='term-section'>[ TESTA A TESTA ]</span>\n"
    metrics_cfg = [("Tiri Porta", 'Shots'), ("Corner", 'Corn'), ("Falli", 'Fouls'), ("Cartellini", 'Cards')]
    for label, key in metrics_cfg:
        ph, pa = calcola_h2h_favorito(stats[key][0], stats[key][1])
        if ph > pa:
            fav_str = f"CASA ({ph*100:.0f}%)"
            if ph >= min_prob: fav_str = f"<span class='term-green'>{fav_str}</span>"
            else: fav_str = f"<span class='term-dim'>{fav_str}</span>"
        else:
            fav_str = f"OSP ({pa*100:.0f}%)"
            if pa >= min_prob: fav_str = f"<span class='term-green'>{fav_str}</span>"
            else: fav_str = f"<span class='term-dim'>{fav_str}</span>"
        html += f"{label:<12} : {fav_str}\n"

    prop_configs = [
        ("CORNER", stats['Corn'][0], stats['Corn'][1], [3.5, 4.5, 5.5], [2.5, 3.5, 4.5], [8.5, 9.5, 10.5]),
        ("TIRI PORTA", stats['Shots'][0], stats['Shots'][1], [3.5, 4.5, 5.5], [2.5, 3.5, 4.5], [7.5, 8.5, 9.5]),
        ("FALLI", stats['Fouls'][0], stats['Fouls'][1], [10.5, 11.5, 12.5], [10.5, 11.5, 12.5], [21.5, 22.5, 23.5]),
        ("CARTELLINI", stats['Cards'][0], stats['Cards'][1], [1.5, 2.5], [1.5, 2.5], [3.5, 4.5]),
        ("GOL", lam_h, lam_a, [0.5, 1.5], [0.5, 1.5], [1.5, 2.5, 3.5])
    ]
    for label, exp_h, exp_a, r_h, r_a, r_tot in prop_configs:
        html += f"\n<span class='term-section'>[ {label} ]</span> (Att: {exp_h:.2f} - {exp_a:.2f})\n"
        html += f"{'LINEA':<15} | {'PROB %':<8} | {'QUOTA'}\n"
        html += "-"*40 + "\n"
        def add_rows(prefix, r, exp):
            rows_html = ""
            for l in r:
                p = poisson.sf(int(l), exp)
                q = 1/p if p > 0 else 99
                row_str = f"{prefix+' Ov '+str(l):<15} | {p*100:04.1f}%   | {q:.2f}"
                if p >= min_prob: rows_html += f"<span class='term-green'>{row_str}</span>\n"
                elif p >= min_prob - 0.10: rows_html += f"<span class='term-dim'>{row_str}</span>\n"
            return rows_html
        html += add_rows("CASA", r_h, exp_h)
        html += add_rows("OSP", r_a, exp_a)
        html += add_rows("TOT", r_tot, exp_h+exp_a)

    html += "</div>"
    return html

# ==============================================================================
# 3. INTERFACCIA
# ==============================================================================

with st.sidebar:
    st.header("🇪🇺 Europe Night")
    api_key_input = st.text_input("API Key", type="password")
    bankroll_input = st.number_input("Bankroll (€)", min_value=10.0, value=26.50, step=0.5)
    
    st.divider()
    min_prob_val = st.slider("Probabilità Minima", 0.50, 0.90, 0.65, step=0.05)
    
    st.divider()
    st.markdown("### 🗺️ Selezione Aree")
    
    selected_groups = []
    for group_name, leagues in LEAGUE_GROUPS.items():
        if st.checkbox(group_name, value=(group_name == "🇪🇺 Coppe Europee")):
            selected_groups.extend(leagues)
            
    st.caption(f"Leghe attive: {len(selected_groups)}")
    show_mapping_errors = st.checkbox("🛠️ Debug Mapping", value=False)

st.title("SmartBet Total Europe")
st.caption("Real Form Engine Attivo (H+A Analysis)")

start_analisys = st.button("🚀 CERCA VALUE BETS", type="primary", use_container_width=True)

if start_analisys:
    if not api_key_input: st.error("Inserisci API Key!")
    elif not selected_groups: st.error("Seleziona almeno un gruppo di leghe!")
    else:
        results_by_league = {}
        global_calendar_data = [] 
        missing_teams_log = [] 
        
        # 1. CARICAMENTO DATI DOMESTICI
        domestic_cache = {}
        leagues_to_load = [k for k in ALL_LEAGUES.keys() if k not in ['UCL','UEL','UECL']]
        
        status = st.empty()
        status.text("Caricamento database completi (potrebbe richiedere 30s)...")
        
        for idx, code in enumerate(leagues_to_load):
            domestic_cache[code] = scarica_dati(code)
            
        # 2. SCANSIONE MATCH
        progress = st.progress(0)
        step = 0
        total_steps = len(selected_groups)
        
        for code in selected_groups:
            league_name = ALL_LEAGUES.get(code, code)
            status.text(f"Analisi: {league_name}...")
            
            if league_name not in results_by_league: results_by_league[league_name] = []
            
            matches = get_live_matches(api_key_input, API_MAPPING.get(code, ''))
            
            if matches:
                for m in matches:
                    if 'home_team' not in m: continue
                    
                    h_raw, a_raw = m['home_team'], m['away_team']
                    h_team = TEAM_MAPPING.get(h_raw, h_raw)
                    a_team = TEAM_MAPPING.get(a_raw, a_raw)
                    
                    raw_date_obj, fmt_date_str = parse_date(m.get('commence_time', ''))
                    
                    # LOGICA CROSS-SEARCH
                    h_stats, h_coeff, h_league = find_team_stats_global(h_team, domestic_cache)
                    a_stats, a_coeff, a_league = find_team_stats_global(a_team, domestic_cache)
                    
                    q1_b, qX_b, q2_b = 0,0,0
                    for b in m['bookmakers']:
                        for mk in b['markets']:
                            if mk['key'] == 'h2h':
                                for o in mk['outcomes']:
                                    if o['name'] == h_raw: q1_b = o['price']
                                    elif o['name'] == 'Draw': qX_b = o['price']
                                    elif o['name'] == a_raw: q2_b = o['price']
                    
                    # SE MANCANO DATI
                    if h_stats is None or a_stats is None:
                        if h_stats is None: missing_teams_log.append(f"LEGA {code}: '{h_raw}' -> Dati mancanti")
                        if a_stats is None: missing_teams_log.append(f"LEGA {code}: '{a_raw}' -> Dati mancanti")
                        
                        html_err = generate_missing_data_terminal(h_team, a_team, (h_stats is not None), (a_stats is not None), {'1':q1_b,'X':qX_b,'2':q2_b})
                        item = {'label': f"⚠️ {fmt_date_str} | {h_team} vs {a_team}", 'html': html_err}
                        results_by_league[league_name].append(item)
                        global_calendar_data.append({'date': raw_date_obj, 'label': f"[{code}] {h_team} vs {a_team}", 'html': html_err})
                        continue

                    # SE DATI OK (Unpack W_ columns directly)
                    stats_final = {
                        'Shots': (h_stats['W_Shots']*h_coeff, a_stats['W_Shots']*a_coeff),
                        'Corn': (h_stats['W_Corn']*h_coeff, a_stats['W_Corn']*a_coeff),
                        'Fouls': (h_stats['W_Fouls'], a_stats['W_Fouls']),
                        'Cards': (h_stats['W_Cards'], a_stats['W_Cards']),
                    }
                    exp_goals_h = h_stats['W_Goals'] * h_coeff
                    exp_goals_a = a_stats['W_Goals'] * a_coeff
                    
                    if q1_b == 0: continue
                    
                    q1_m, qX_m, q2_m, lam_h, lam_a = calcola_1x2_lambda(exp_goals_h, exp_goals_a)
                    roi_1 = ((1/q1_m)*q1_b)-1; roi_X = ((1/qX_m)*qX_b)-1; roi_2 = ((1/q2_m)*q2_b)-1
                    
                    html_block = generate_complete_terminal(
                        h_team, a_team, stats_final, lam_h, lam_a, 
                        {'1':q1_b,'X':qX_b,'2':q2_b}, {'1':roi_1,'X':roi_X,'2':roi_2},
                        min_prob_val, h_stats['Date'], a_stats['Date']
                    )
                    
                    item = {'label': f"✅ {fmt_date_str} | {h_team} vs {a_team}", 'html': html_block}
                    results_by_league[league_name].append(item)
                    global_calendar_data.append({'date': raw_date_obj, 'label': f"[{code}] {h_team} vs {a_team}", 'html': html_block})

            step += 1
            progress.progress(step / total_steps)
            
        status.empty()
        
        if show_mapping_errors and missing_teams_log:
            unique_errors = sorted(list(set(missing_teams_log)))
            st.warning(f"⚠️ Debug: {len(unique_errors)} squadre non trovate.")
            st.text_area("📋 Copia questa lista:", value="\n".join(unique_errors), height=400)
            
        st.success("Analisi Europea Completata.")
        
        main_tab1, main_tab2 = st.tabs(["🏆 COMPETIZIONI", "📅 CALENDARIO"])
        with main_tab1:
            active_leagues = [l for l in results_by_league.keys() if results_by_league[l]]
            if active_leagues:
                league_tabs = st.tabs(active_leagues)
                for i, lname in enumerate(active_leagues):
                    with league_tabs[i]:
                        for m in results_by_league[lname]:
                            with st.expander(m['label']): st.markdown(m['html'], unsafe_allow_html=True)
            else:
                st.write("Nessun risultato trovato.")
        with main_tab2:
            st.markdown("#### Seleziona Data")
            if global_calendar_data:
                dates = sorted(list(set([d['date'] for d in global_calendar_data])))
                if dates:
                    sel_d = st.date_input("Giorno:", value=dates[0], min_value=dates[0])
                    filtered = [m for m in global_calendar_data if m['date'] == sel_d]
                    if filtered:
                        for m in filtered:
                            with st.expander(m['label']): st.markdown(m['html'], unsafe_allow_html=True)
                    else: st.warning("Nessuna partita.")
            else: st.write("Nessun dato.")
