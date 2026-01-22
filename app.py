import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from datetime import datetime, timedelta

# ==============================================================================
# 1. CONFIGURAZIONE E COSTANTI
# ==============================================================================
st.set_page_config(page_title="SmartBet Europe", page_icon="🇪🇺", layout="centered")

# COSTANTI GLOBALI
STAGIONE = "2526"
REGION = 'eu'
MARKET = 'h2h'

# CSS Custom
st.markdown("""
<style>
    .stProgress { display: none; }
    .terminal-box {
        font-family: "Courier New", Courier, monospace;
        background-color: #0c0c0c;
        color: #cccccc;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #333;
        white-space: pre; 
        overflow-x: auto;
        font-size: 0.9em;
        margin-bottom: 10px;
    }
    .term-header { color: #FFD700; font-weight: bold; } 
    .term-section { color: #00FFFF; font-weight: bold; margin-top: 10px; display: block; } 
    .term-green { color: #00FF00; font-weight: bold; } 
    .term-val { color: #FF00FF; font-weight: bold; }
    .term-warn { color: #FF4500; font-weight: bold; background-color: #330000; padding: 2px; }
    .term-dim { color: #555555; }
    .streamlit-expanderHeader { font-weight: bold; background-color: #f0f2f6; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# DATABASE LEGHE (AGGIUNTE LE COPPE)
ALL_LEAGUES = {
    # COPPE EUROPEE (Speciali)
    'UCL': '🇪🇺 UEFA Champions League',
    'UEL': '🇪🇺 UEFA Europa League',
    'UECL': '🇪🇺 UEFA Conference League',
    
    # CAMPIONATI DOMESTICI (Dove cerchiamo i dati)
    'I1': '🇮🇹 ITA - Serie A', 'E0': '🇬🇧 ENG - Premier League', 'SP1': '🇪🇸 ESP - La Liga',
    'D1': '🇩🇪 GER - Bundesliga', 'F1': '🇫🇷 FRA - Ligue 1', 
    'I2': '🇮🇹 ITA - Serie B', 'E1': '🇬🇧 ENG - Championship', 'E2': '🇬🇧 ENG - League One',
    'N1': '🇳🇱 NED - Eredivisie', 'P1': '🇵🇹 POR - Primeira Liga',
    'B1': '🇧🇪 BEL - Pro League', 'T1': '🇹🇷 TUR - Super Lig'
}

API_MAPPING = {
    # API Coppe
    'UCL': 'soccer_uefa_champs_league',
    'UEL': 'soccer_uefa_europa_league',
    'UECL': 'soccer_uefa_euro_conference',
    
    # API Domestiche
    'I1': 'soccer_italy_serie_a', 'I2': 'soccer_italy_serie_b',
    'E0': 'soccer_epl', 'E1': 'soccer_efl_champ', 'E2': 'soccer_england_league1',
    'SP1': 'soccer_spain_la_liga', 'D1': 'soccer_germany_bundesliga',
    'F1': 'soccer_france_ligue_one', 'N1': 'soccer_netherlands_eredivisie',
    'P1': 'soccer_portugal_primeira_liga', 'B1': 'soccer_belgium_pro_league',
    'T1': 'soccer_turkey_super_league'
}

# COEFFICIENTI DI DIFFICOLTA' (Per normalizzare stats tra campionati diversi)
LEAGUE_COEFF = {
    'E0': 1.00, 'SP1': 0.95, 'I1': 0.92, 'D1': 0.90, 'F1': 0.85, # Top 5
    'P1': 0.80, 'N1': 0.78, 'E1': 0.75, # Tier 2 Strong
    'B1': 0.70, 'T1': 0.65, 'I2': 0.60, 'E2': 0.55 # Tier 2/3
}

# MEGA MAPPING
TEAM_MAPPING = {
    'Inter Milan': 'Inter', 'AC Milan': 'Milan', 'Juventus': 'Juve', 'Napoli': 'Napoli', 
    'Roma': 'Roma', 'Lazio': 'Lazio', 'Atalanta BC': 'Atalanta', 'Hellas Verona': 'Verona',
    'Udinese Calcio': 'Udinese', 'Cagliari Calcio': 'Cagliari', 'US Lecce': 'Lecce', 
    'Empoli FC': 'Empoli', 'Sassuolo Calcio': 'Sassuolo', 'Salernitana': 'Salernitana', 
    'Monza': 'Monza', 'Frosinone': 'Frosinone', 'Genoa': 'Genoa', 'Parma': 'Parma', 
    'Como': 'Como', 'Venezia': 'Venezia', 'Pisa': 'Pisa', 'Cremonese': 'Cremonese',
    'Palermo': 'Palermo', 'Bari': 'Bari', 'Sampdoria': 'Sampdoria', 'Spezia Calcio': 'Spezia',
    'Modena FC': 'Modena', 'Catanzaro': 'Catanzaro', 'Reggiana': 'Reggiana', 'Brescia': 'Brescia',
    'Cosenza': 'Cosenza', 'Sudtirol': 'Sudtirol', 'Cittadella': 'Cittadella', 'Mantova': 'Mantova',
    'Cesena': 'Cesena', 'Juve Stabia': 'Juve Stabia', 'Carrarese': 'Carrarese',
    'Manchester United': 'Man United', 'Manchester City': 'Man City', 'Tottenham Hotspur': 'Tottenham',
    'Newcastle United': 'Newcastle', 'Wolverhampton Wanderers': 'Wolves', 'Brighton and Hove Albion': 'Brighton',
    'West Ham United': 'West Ham', 'Leeds United': 'Leeds', 'Nottingham Forest': "Nott'm Forest",
    'Leicester City': 'Leicester', 'Norwich City': 'Norwich', 'Sheffield United': 'Sheffield United',
    'Blackburn Rovers': 'Blackburn', 'West Bromwich Albion': 'West Brom', 'Coventry City': 'Coventry',
    'Middlesbrough': 'Middlesbrough', 'Stoke City': 'Stoke', 'Queens Park Rangers': 'QPR',
    'Preston North End': 'Preston', 'Sheffield Wednesday': 'Sheffield Weds', 'Luton Town': 'Luton',
    'Burnley': 'Burnley', 'Watford': 'Watford', 'Sunderland AFC': 'Sunderland', 'Sunderland': 'Sunderland',
    'Derby County': 'Derby', 'Birmingham City': 'Birmingham', 'Swansea City': 'Swansea',
    'Wrexham AFC': 'Wrexham', 'Oxford United': 'Oxford', 'Charlton Athletic': 'Charlton',
    'Ipswich Town': 'Ipswich', 'Hull City': 'Hull', 'Bristol City': 'Bristol City', 
    'Cardiff City': 'Cardiff', 'Portsmouth': 'Portsmouth', 'Plymouth Argyle': 'Plymouth', 'Millwall': 'Millwall',
    'Lincoln City': 'Lincoln', 'Bolton Wanderers': 'Bolton', 'Huddersfield Town': 'Huddersfield',
    'Atletico Madrid': 'Ath Madrid', 'Athletic Bilbao': 'Ath Bilbao', 'Real Betis': 'Betis',
    'Real Sociedad': 'Sociedad', 'Rayo Vallecano': 'Vallecano', 'Celta Vigo': 'Celta', 
    'Alavés': 'Alaves', 'Cadiz CF': 'Cadiz', 'UD Las Palmas': 'Las Palmas', 'RCD Espanyol': 'Espanyol',
    'Real Valladolid': 'Valladolid', 'Leganés': 'Leganes', 'Girona FC': 'Girona',
    'Bayern Munich': 'Bayern Munich', 'Bayer Leverkusen': 'Leverkusen', 'Borussia Dortmund': 'Dortmund',
    'Borussia Monchengladbach': "M'gladbach", 'Eintracht Frankfurt': 'Ein Frankfurt', 
    'SC Freiburg': 'Freiburg', '1. FC Köln': 'FC Koln', 'Mainz 05': 'Mainz', 'VfL Bochum': 'Bochum',
    'VfB Stuttgart': 'Stuttgart', 'FC St. Pauli': 'St Pauli', 'Holstein Kiel': 'Holstein Kiel',
    'TSG Hoffenheim': 'Hoffenheim', 'Werder Bremen': 'Werder Bremen', 'Augsburg': 'Augsburg',
    'Paris Saint Germain': 'Paris SG', 'Marseille': 'Marseille', 'Saint-Etienne': 'St Etienne',
    'Clermont Foot': 'Clermont', 'Le Havre': 'Le Havre', 'RC Lens': 'Lens', 'AS Monaco': 'Monaco',
    'Lille OSC': 'Lille', 'Olympique Lyonnais': 'Lyon', 'Brest': 'Brest', 'Stade Brestois': 'Brest',
    'PSV Eindhoven': 'PSV Eindhoven', 'Feyenoord Rotterdam': 'Feyenoord', 'Ajax Amsterdam': 'Ajax',
    'AZ Alkmaar': 'AZ Alkmaar', 'FC Twente': 'Twente', 'FC Utrecht': 'Utrecht',
    'Sparta Rotterdam': 'Sparta Rotterdam', 'NEC Nijmegen': 'Nijmegen', 'Go Ahead Eagles': 'Go Ahead Eagles',
    'Fortuna Sittard': 'For Sittard', 'PEC Zwolle': 'Zwolle', 'Almere City': 'Almere City',
    'RKC Waalwijk': 'Waalwijk', 'SC Heerenveen': 'Heerenveen', 'Heracles Almelo': 'Heracles',
    'Sporting CP': 'Sp Lisbon', 'Sporting Lisbon': 'Sp Lisbon', 
    'Benfica': 'Benfica', 'FC Porto': 'Porto', 'Sporting Braga': 'Braga', 'Vitoria Guimaraes': 'Guimaraes',
    'Boavista FC': 'Boavista', 'Estoril Praia': 'Estoril', 'Casa Pia AC': 'Casa Pia',
    'Farense': 'Farense', 'Arouca': 'Arouca', 'Gil Vicente': 'Gil Vicente',
    'Galatasaray': 'Galatasaray', 'Fenerbahce': 'Fenerbahce', 'Besiktas': 'Besiktas', 
    'Trabzonspor': 'Trabzonspor', 'Istanbul Basaksehir': 'Basaksehir', 'Samsunspor': 'Samsunspor',
    'Kasimpasa': 'Kasimpasa', 'Alanyaspor': 'Alanyaspor', 'Antalyaspor': 'Antalyaspor'
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
    # Se è una coppa, non c'è CSV da scaricare
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
            
        # CALCOLI BASE (Weighted)
        df['W_Goals'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.ewm(span=5).mean())
        df['W_ST'] = df.groupby('HomeTeam')['HST'].transform(lambda x: x.ewm(span=5).mean())
        df['W_C'] = df.groupby('HomeTeam')['HC'].transform(lambda x: x.ewm(span=5).mean())
        df['W_F'] = df.groupby('HomeTeam')['HF'].transform(lambda x: x.ewm(span=5).mean())
        df['W_Y'] = df.groupby('HomeTeam')['HY'].transform(lambda x: x.ewm(span=5).mean())
        
        # Semplificazione: usiamo un DF verticale per ricerca rapida
        home_df = df[['Date','HomeTeam','W_Goals','W_ST','W_C','W_F','W_Y']].rename(columns={'HomeTeam':'Team'})
        away_df = df[['Date','AwayTeam','W_Goals','W_ST','W_C','W_F','W_Y']].rename(columns={'AwayTeam':'Team'}) # Approx
        
        full_df = pd.concat([home_df, away_df]).sort_values(['Team','Date'])
        return full_df, df 
    except: return None, None

def get_live_matches(api_key, sport_key):
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions={REGION}&markets={MARKET}'
    try: return requests.get(url).json()
    except: return []

def calcola_1x2_lambda(exp_shots_h, exp_shots_a):
    lam_h = exp_shots_h * 0.30 if exp_shots_h > 0 else 1.0
    lam_a = exp_shots_a * 0.30 if exp_shots_a > 0 else 0.8
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
    mat = mat / np.sum(mat)
    
    p1 = np.sum(np.tril(mat,-1)); pX = np.trace(mat); p2 = np.sum(np.triu(mat,1))
    return (1/p1 if p1>0 else 99), (1/pX if pX>0 else 99), (1/p2 if p2>0 else 99), lam_h, lam_a

def calcola_h2h_favorito(val_h, val_a):
    r = np.arange(40)
    pmf_h = poisson.pmf(r, val_h); pmf_a = poisson.pmf(r, val_a)
    joint = np.outer(pmf_h, pmf_a)
    p_h = np.sum(np.tril(joint, -1)); p_a = np.sum(np.triu(joint, 1))
    return p_h, p_a

# --- NUOVO: RICERCA GLOBALE PER COPPE ---
# Cerca il team in TUTTI i dataframe caricati
def find_team_stats_global(team_name, cache_dataframes):
    for league_code, (df_weighted, _) in cache_dataframes.items():
        if df_weighted is None: continue
        
        # Cerca nel DF verticale
        team_stats = df_weighted[df_weighted['Team'] == team_name]
        
        if not team_stats.empty:
            last_row = team_stats.iloc[-1]
            coeff = LEAGUE_COEFF.get(league_code, 0.70) # Coefficiente Lega
            return last_row, coeff
            
    return None, 0

def generate_complete_terminal(h_team, a_team, stats, lam_h, lam_a, odds_1x2, roi_1x2, min_prob, last_date_h, last_date_a):
    html = f"""<div class='terminal-box'>"""
    
    # DATA CHECK
    max_date = max(last_date_h, last_date_a)
    days_lag = (datetime.now() - max_date).days
    
    if days_lag > 14:
        html += f"<div class='term-warn'>⚠️ DATI DATATI ({days_lag}gg). FORMA NON ATTENDIBILE.</div>\n"
    elif days_lag > 7:
        html += f"<div style='color:orange;'>⚠️ Dati a {days_lag} giorni fa</div>\n"
    
    # 1X2
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

    # PROPS
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
    st.header("🇪🇺 Europe Setup")
    api_key_input = st.text_input("API Key", type="password")
    bankroll_input = st.number_input("Bankroll (€)", min_value=10.0, value=26.50, step=0.5)
    
    st.divider()
    min_prob_val = st.slider("Probabilità Minima", 0.50, 0.90, 0.65, step=0.05)
    
    st.divider()
    # PRE-SELEZIONE ANCHE DELLE COPPE
    default_leagues = ['I1', 'E0', 'SP1', 'D1', 'F1', 'UCL', 'UEL', 'UECL']
    selected_leagues_keys = st.multiselect("Leghe & Coppe:", options=list(ALL_LEAGUES.keys()), format_func=lambda x: ALL_LEAGUES[x], default=default_leagues)
    
    show_mapping_errors = st.checkbox("🛠️ Mostra Errori Mapping", value=False)

st.title("🇪🇺 SmartBet European Night")
st.caption("Cross-League Engine Active: Ranking UEFA Applied")

start_analisys = st.button("🚀 CERCA VALUE BETS", type="primary", use_container_width=True)

if start_analisys:
    if not api_key_input: st.error("Inserisci API Key!")
    elif not selected_leagues_keys: st.error("Seleziona competizioni!")
    else:
        results_by_league = {ALL_LEAGUES[k]: [] for k in selected_leagues_keys}
        global_calendar_data = [] 
        missing_teams_log = [] 
        
        # 1. CARICAMENTO DATI DOMESTICI (CACHE)
        # Scarichiamo prima TUTTI i campionati domestici necessari per il lookup delle coppe
        domestic_cache = {}
        # Identifichiamo quali leghe servono (se ho selezionato UCL, devo caricare I1, E0, ecc per cercare i team)
        # Per sicurezza, carichiamo tutte le leghe domestiche note in ALL_LEAGUES tranne le coppe
        leagues_to_load = [k for k in ALL_LEAGUES.keys() if k not in ['UCL','UEL','UECL']]
        
        status = st.empty()
        status.text("Caricamento database domestici...")
        for code in leagues_to_load:
            domestic_cache[code] = scarica_dati(code)
            
        # 2. SCANSIONE MATCH
        progress = st.progress(0)
        step = 0
        total_steps = len(selected_leagues_keys)
        
        for code in selected_leagues_keys:
            league_name = ALL_LEAGUES[code]
            status.text(f"Analisi: {league_name}...")
            
            # Se è una coppa o lega normale, l'API call è la stessa logica
            matches = get_live_matches(api_key_input, API_MAPPING.get(code, ''))
            
            if matches:
                for m in matches:
                    if 'home_team' not in m: continue
                    
                    h_raw, a_raw = m['home_team'], m['away_team']
                    h_team = TEAM_MAPPING.get(h_raw, h_raw)
                    a_team = TEAM_MAPPING.get(a_raw, a_raw)
                    
                    raw_date_obj, fmt_date_str = parse_date(m.get('commence_time', ''))
                    
                    # LOGICA CROSS-SEARCH
                    # Cerchiamo le stats in domestic_cache
                    h_stats, h_coeff = find_team_stats_global(h_team, domestic_cache)
                    a_stats, a_coeff = find_team_stats_global(a_team, domestic_cache)
                    
                    if h_stats is None:
                        missing_teams_log.append(f"LEGA {code}: '{h_raw}' -> Dati storici non trovati")
                        continue
                    if a_stats is None:
                        missing_teams_log.append(f"LEGA {code}: '{a_raw}' -> Dati storici non trovati")
                        continue
                        
                    # NORMALIZZAZIONE (APPLICAZIONE COEFFICIENTE)
                    # Stats * Coeff = Adjusted Stats per l'Europa
                    # Es. Goal in Premier (1.0) valgono interi. Goal in Turchia (0.65) valgono meno.
                    stats_final = {
                        'Shots': (h_stats['W_ST']*h_coeff, a_stats['W_ST']*a_coeff),
                        'Corn': (h_stats['W_C']*h_coeff, a_stats['W_C']*a_coeff),
                        'Fouls': (h_stats['W_F'], a_stats['W_F']), # Falli di solito non si scalano
                        'Cards': (h_stats['W_Y'], a_stats['W_Y']),
                    }
                    exp_goals_h = h_stats['W_Goals'] * h_coeff
                    exp_goals_a = a_stats['W_Goals'] * a_coeff
                    
                    # BOOKIE ODDS
                    q1_b, qX_b, q2_b = 0,0,0
                    for b in m['bookmakers']:
                        for mk in b['markets']:
                            if mk['key'] == 'h2h':
                                for o in mk['outcomes']:
                                    if o['name'] == h_raw: q1_b = o['price']
                                    elif o['name'] == 'Draw': qX_b = o['price']
                                    elif o['name'] == a_raw: q2_b = o['price']
                    if q1_b == 0: continue
                    
                    # CALCOLO PROBABILITA'
                    q1_m, qX_m, q2_m, lam_h, lam_a = calcola_1x2_lambda(exp_goals_h, exp_goals_a)
                    roi_1 = ((1/q1_m)*q1_b)-1; roi_X = ((1/qX_m)*qX_b)-1; roi_2 = ((1/q2_m)*q2_b)-1
                    
                    html_block = generate_complete_terminal(
                        h_team, a_team, stats_final, lam_h, lam_a, 
                        {'1':q1_b,'X':qX_b,'2':q2_b}, {'1':roi_1,'X':roi_X,'2':roi_2},
                        min_prob_val, h_stats['Date'], a_stats['Date']
                    )
                    
                    item = {'label': f"{fmt_date_str} | {h_team} vs {a_team}", 'html': html_block}
                    results_by_league[league_name].append(item)
                    global_calendar_data.append({'date': raw_date_obj, 'label': f"[{code}] {h_team} vs {a_team}", 'html': html_block})

            step += 1
            progress.progress(step / total_steps)
            
        status.empty()
        
        if show_mapping_errors and missing_teams_log:
            st.warning(f"⚠️ {len(missing_teams_log)} SQUADRE NON TROVATE (Possibile mancanza dati CSV o Mapping Errato)")
            for err in list(set(missing_teams_log)): st.error(err)
            
        st.success("Analisi Europea Completata.")
        
        main_tab1, main_tab2 = st.tabs(["🏆 COMPETIZIONI", "📅 CALENDARIO"])
        with main_tab1:
            league_tabs = st.tabs([ALL_LEAGUES[k] for k in selected_leagues_keys])
            for i, code in enumerate(selected_leagues_keys):
                with league_tabs[i]:
                    matches = results_by_league[ALL_LEAGUES[code]]
                    if matches:
                        for m in matches:
                            with st.expander(m['label']): st.markdown(m['html'], unsafe_allow_html=True)
                    else: st.write("Nessun match o dati insufficienti.")
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
