import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

# Configurazione Pagina
st.set_page_config(page_title="SmartBet Terminal", page_icon="📟", layout="centered")

# CSS Custom per lo stile Terminale
st.markdown("""
<style>
    /* Nasconde elementi inutili */
    .stProgress { display: none; }
    
    /* Stile Matrix per i blocchi */
    .terminal-box {
        font-family: "Courier New", Courier, monospace;
        background-color: #0c0c0c;
        color: #cccccc;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #333;
        margin-bottom: 20px;
        white-space: pre; /* Mantiene formattazione spazi */
        overflow-x: auto;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Colori del terminale */
    .term-header { color: #FFD700; font-weight: bold; font-size: 1.1em; } /* Oro */
    .term-section { color: #00FFFF; font-weight: bold; margin-top: 10px; } /* Ciano */
    .term-green { color: #00FF00; font-weight: bold; } /* Verde Matrix */
    .term-val { color: #FF00FF; font-weight: bold; } /* Magenta per Value */
    .term-label { color: #aaaaaa; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Setup")
    api_key_input = st.text_input("API Key", type="password")
    bankroll_input = st.number_input("Bankroll (€)", min_value=10.0, value=26.50, step=0.5)
    st.info("v34.0 - Full Terminal Mode")

st.title("📟 SmartBet AI Terminal")
st.caption(f"Bankroll Attuale: €{bankroll_input:.2f}")

start_analisys = st.button("🚀 CERCA VALUE BETS", type="primary", use_container_width=True)

# DATI COSTANTI
STAGIONE = "2526"
REGION = 'eu'
MARKET = 'h2h'
LEGHE = {'I1': '🇮🇹 ITA', 'E0': '🇬🇧 ENG', 'SP1': '🇪🇸 ESP', 'D1': '🇩🇪 GER', 'F1': '🇫🇷 FRA'}

TEAM_MAPPING = {
    'AC Milan': 'Milan', 'Inter Milan': 'Inter', 'Internazionale': 'Inter', 'AS Roma': 'Roma', 
    'Atalanta BC': 'Atalanta', 'Hellas Verona': 'Verona', 'Udinese Calcio': 'Udinese', 
    'Cagliari Calcio': 'Cagliari', 'US Lecce': 'Lecce', 'Empoli FC': 'Empoli', 
    'Sassuolo Calcio': 'Sassuolo', 'Salernitana': 'Salernitana', 'Monza': 'Monza', 
    'Frosinone': 'Frosinone', 'Genoa': 'Genoa', 'Parma': 'Parma', 'Como': 'Como', 
    'Venezia': 'Venezia', 'Pisa': 'Pisa', 'Cremonese': 'Cremonese', 'SSC Napoli': 'Napoli', 
    'SS Lazio': 'Lazio', 'Bologna FC': 'Bologna', 'Torino': 'Torino', 'Fiorentina': 'Fiorentina', 
    'Spezia Calcio': 'Spezia', 'Sampdoria': 'Sampdoria', 'Manchester United': 'Man United', 
    'Manchester City': 'Man City', 'Man City': 'Man City', 'Tottenham Hotspur': 'Tottenham', 
    'Tottenham': 'Tottenham', 'Newcastle United': 'Newcastle', 'Wolverhampton Wanderers': 'Wolves', 
    'Brighton and Hove Albion': 'Brighton', 'West Ham United': 'West Ham', 'Leeds United': 'Leeds', 
    'Nottingham Forest': "Nott'm Forest", 'Leicester City': 'Leicester', 'Norwich City': 'Norwich', 
    'Watford': 'Watford', 'Brentford': 'Brentford', 'Crystal Palace': 'Crystal Palace', 
    'Southampton': 'Southampton', 'Everton': 'Everton', 'Aston Villa': 'Aston Villa', 
    'Liverpool': 'Liverpool', 'Chelsea': 'Chelsea', 'Arsenal': 'Arsenal', 'Fulham': 'Fulham', 
    'Bournemouth': 'Bournemouth', 'Sheffield United': 'Sheffield United', 'Luton Town': 'Luton', 
    'Burnley': 'Burnley', 'Ipswich Town': 'Ipswich', 'Real Madrid': 'Real Madrid', 
    'FC Barcelona': 'Barcelona', 'Barcelona': 'Barcelona', 'Atletico Madrid': 'Ath Madrid', 
    'Atlético Madrid': 'Ath Madrid', 'Athletic Bilbao': 'Ath Bilbao', 'Athletic Club': 'Ath Bilbao', 
    'Real Betis': 'Betis', 'Real Sociedad': 'Sociedad', 'Villarreal CF': 'Villarreal', 
    'Sevilla FC': 'Sevilla', 'Sevilla': 'Sevilla', 'Valencia CF': 'Valencia', 'CA Osasuna': 'Osasuna', 
    'Rayo Vallecano': 'Vallecano', 'RCD Mallorca': 'Mallorca', 'Mallorca': 'Mallorca', 
    'Celta Vigo': 'Celta', 'RC Celta': 'Celta', 'Girona FC': 'Girona', 'Getafe CF': 'Getafe', 
    'UD Almería': 'Almeria', 'Alavés': 'Alaves', 'Deportivo Alavés': 'Alaves', 'Cadiz CF': 'Cadiz', 
    'Granada CF': 'Granada', 'UD Las Palmas': 'Las Palmas', 'Elche CF': 'Elche', 
    'RCD Espanyol': 'Espanyol', 'Espanyol': 'Espanyol', 'Real Valladolid': 'Valladolid', 
    'Leganés': 'Leganes', 'Levante': 'Levante', 'Eibar': 'Eibar', 'Oviedo': 'Oviedo', 
    'Bayern Munich': 'Bayern Munich', 'Borussia Dortmund': 'Dortmund', 'Bayer Leverkusen': 'Leverkusen', 
    'RB Leipzig': 'RB Leipzig', '1. FC Union Berlin': 'Union Berlin', 'Union Berlin': 'Union Berlin', 
    'SC Freiburg': 'Freiburg', 'Eintracht Frankfurt': 'Ein Frankfurt', 'VfL Wolfsburg': 'Wolfsburg', 
    'FSV Mainz 05': 'Mainz', 'Borussia Monchengladbach': "M'gladbach", 
    'Borussia Mönchengladbach': "M'gladbach", '1. FC Köln': 'FC Koln', 'FC Koln': 'FC Koln', 
    'TSG Hoffenheim': 'Hoffenheim', 'SV Werder Bremen': 'Werder Bremen', 'Werder Bremen': 'Werder Bremen', 
    'VfL Bochum': 'Bochum', 'FC Augsburg': 'Augsburg', 'Augsburg': 'Augsburg', 
    'VfB Stuttgart': 'Stuttgart', '1. FC Heidenheim': 'Heidenheim', 'Darmstadt 98': 'Darmstadt', 
    'FC St. Pauli': 'St Pauli', 'St. Pauli': 'St Pauli', 'Holstein Kiel': 'Holstein Kiel', 
    'Hamburger SV': 'Hamburg', 'Paris Saint Germain': 'Paris SG', 'PSG': 'Paris SG', 
    'Marseille': 'Marseille', 'Olympique Marseille': 'Marseille', 'AS Monaco': 'Monaco', 
    'Monaco': 'Monaco', 'Lyon': 'Lyon', 'Olympique Lyonnais': 'Lyon', 'LOSC Lille': 'Lille', 
    'Lille OSC': 'Lille', 'Stade Rennais': 'Rennes', 'Rennes': 'Rennes', 'Nice': 'Nice', 
    'OGC Nice': 'Nice', 'RC Lens': 'Lens', 'Lens': 'Lens', 'Reims': 'Reims', 
    'Stade de Reims': 'Reims', 'Montpellier': 'Montpellier', 'Toulouse FC': 'Toulouse', 
    'Strasbourg': 'Strasbourg', 'Nantes': 'Nantes', 'Lorient': 'Lorient', 
    'Clermont Foot': 'Clermont', 'Metz': 'Metz', 'Le Havre': 'Le Havre', 'Brest': 'Brest', 
    'Stade Brestois': 'Brest', 'Saint-Etienne': 'St Etienne', 'ASSE': 'St Etienne', 
    'Auxerre': 'Auxerre', 'Angers SCO': 'Angers'
}

@st.cache_data(ttl=3600)
def scarica_dati(codice_lega):
    url = f"https://www.football-data.co.uk/mmz4281/{STAGIONE}/{codice_lega}.csv"
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date')
        df['HomeTeam'] = df['HomeTeam'].str.strip(); df['AwayTeam'] = df['AwayTeam'].str.strip()
        
        df_temp = pd.concat([
            df[['Date','HomeTeam','AST']].rename(columns={'HomeTeam':'Team','AST':'Conceded'}),
            df[['Date','AwayTeam','HST']].rename(columns={'AwayTeam':'Team','HST':'Conceded'})
        ]).sort_values(['Team','Date'])
        
        defense = df_temp.groupby('Team')['Conceded'].mean().to_dict()
        avg_conc = df_temp['Conceded'].mean()
        def get_diff(opp): return avg_conc / defense.get(opp, avg_conc) if defense.get(opp, avg_conc) > 0 else 1.0

        cols_h = {'HST':'ShotsFor','AST':'ShotsAg','HC':'CornFor','AC':'CornAg','HF':'FoulsFor','AF':'FoulsAg','HY':'CardsFor','AY':'CardsAg'}
        home = df[['Date','HomeTeam', 'AwayTeam'] + list(cols_h.keys())].rename(columns={'HomeTeam':'Team', 'AwayTeam':'Opponent', **cols_h}); home['IsHome'] = 1
        cols_a = {'AST':'ShotsFor','HST':'ShotsAg','AC':'CornFor','HC':'CornAg','AF':'FoulsFor','HF':'FoulsAg','AY':'CardsFor','HY':'CardsAg'}
        away = df[['Date','AwayTeam', 'HomeTeam'] + list(cols_a.keys())].rename(columns={'AwayTeam':'Team', 'HomeTeam':'Opponent', **cols_a}); away['IsHome'] = 0
        
        df_teams = pd.concat([home, away]).sort_values(['Team', 'Date'])
        df_teams['Diff_Factor'] = df_teams['Opponent'].map(lambda x: get_diff(x))
        
        for m in ['ShotsFor','ShotsAg','CornFor','CornAg', 'FoulsFor','FoulsAg','CardsFor','CardsAg']:
            val = df_teams[m] * df_teams['Diff_Factor'] if m in ['ShotsFor','ShotsAg','CornFor','CornAg'] else df_teams[m]
            df_teams[f'W_{m}'] = (val.shift(1).rolling(5).mean() * 0.60) + (val.shift(1).expanding().mean() * 0.40)
        return df_teams.dropna(), df
    except: return None, None

def get_live_matches(api_key, sport_key):
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions={REGION}&markets={MARKET}'
    try: return requests.get(url).json()
    except: return []

def calcola_1x2_lambda(exp_shots_h, exp_shots_a):
    lam_h = exp_shots_h * 0.30; lam_a = exp_shots_a * 0.30
    mat = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            mat[i,j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
    if lam_h < 1.3 and lam_a < 1.3: mat[0,0]*=1.15; mat[1,1]*=1.08 
    p1 = np.sum(np.tril(mat,-1)); pX = np.trace(mat); p2 = np.sum(np.triu(mat,1))
    return (1/p1 if p1>0 else 99), (1/pX if pX>0 else 99), (1/p2 if p2>0 else 99), lam_h, lam_a

def calcola_h2h_favorito(val_h, val_a):
    r = np.arange(40)
    pmf_h = poisson.pmf(r, val_h); pmf_a = poisson.pmf(r, val_a)
    joint = np.outer(pmf_h, pmf_a)
    p_h = np.sum(np.tril(joint, -1))
    p_a = np.sum(np.triu(joint, 1))
    return p_h, p_a

def get_full_stats(home, away, df_teams, df_matches):
    try:
        s_h = df_teams[df_teams['Team'] == home].iloc[-1]
        s_a = df_teams[df_teams['Team'] == away].iloc[-1]
    except: return None
    res = {}
    config = [('Shots','HST','AST'), ('Corn','HC','AC'), ('Fouls','HF','AF'), ('Cards','HY','AY')]
    for name, ch, ca in config:
        avg_L_h = df_matches[ch].mean(); avg_L_a = df_matches[ca].mean()
        att_h = s_h[f'W_{name}For'] / avg_L_h; def_a = s_a[f'W_{name}Ag'] / avg_L_h; exp_h = att_h * def_a * avg_L_h
        att_a = s_a[f'W_{name}For'] / avg_L_a; def_h = s_h[f'W_{name}Ag'] / avg_L_a; exp_a = att_a * def_h * avg_L_a
        res[name] = (exp_h, exp_a)
    return res

# --- MOTORE DI GENERAZIONE HTML TERMINALE ---
def generate_complete_terminal(h_team, a_team, stats, lam_h, lam_a, odds_1x2, roi_1x2):
    # Inizio Blocco
    html = f"""<div class='terminal-box'>
<span class='term-header'>{h_team} vs {a_team}</span>
{'='*55}
"""
    # SEZIONE 1: 1X2 E VALORE
    html += f"\n<span class='term-section'>[ 1X2 ANALYSIS ]</span>\n"
    html += f"{'SEGNO':<6} | {'MY QUOTA':<10} | {'BOOKIE':<8} | {'VALUE'}\n"
    html += "-"*45 + "\n"
    
    segni = [('1', roi_1x2['1'], odds_1x2['1']), ('X', roi_1x2['X'], odds_1x2['X']), ('2', roi_1x2['2'], odds_1x2['2'])]
    
    for segno, roi, book_q in segni:
        # Ricavo my quota dal roi: roi = (1/my * book) - 1  => my = book / (roi + 1)
        my_q = book_q / (roi + 1) if (roi+1) > 0 else 99.0
        
        # Colorazione Valore
        val_str = f"{roi*100:+.0f}%"
        if roi >= 0.15 and book_q <= 5.0: # Regola Valore Strict
            val_str = f"<span class='term-val'>{val_str} (TOP)</span>"
        elif roi > 0:
            val_str = f"<span class='term-green'>{val_str}</span>"
        else:
            val_str = f"{val_str}"
            
        html += f"{segno:<6} | {my_q:<10.2f} | {book_q:<8.2f} | {val_str}\n"

    # SEZIONE 2: TESTA A TESTA
    html += f"\n<span class='term-section'>[ TESTA A TESTA (FAVORITO) ]</span>\n"
    metrics_cfg = [("Tiri Porta", 'Shots'), ("Corner", 'Corn'), ("Falli", 'Fouls'), ("Cartellini", 'Cards')]
    for label, key in metrics_cfg:
        ph, pa = calcola_h2h_favorito(stats[key][0], stats[key][1])
        if ph > pa:
            fav_str = f"CASA ({ph*100:.0f}%)"
            if ph > 0.70: fav_str = f"<span class='term-green'>{fav_str}</span>"
        else:
            fav_str = f"OSP ({pa*100:.0f}%)"
            if pa > 0.70: fav_str = f"<span class='term-green'>{fav_str}</span>"
            
        html += f"{label:<12} : {fav_str}\n"

    # SEZIONE 3: PROP BETS (TUTTE)
    # Configurazione Range Completa
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
                if p > 0.70:
                    rows_html += f"<span class='term-green'>{row_str}</span>\n"
                else:
                    rows_html += f"{row_str}\n"
            return rows_html

        html += add_rows("CASA", r_h, exp_h)
        html += add_rows("OSP", r_a, exp_a)
        html += add_rows("TOT", r_tot, exp_h+exp_a)

    html += "</div>"
    return html

# MAIN LOOP
if start_analisys:
    if not api_key_input:
        st.error("Inserisci l'API Key nel menu laterale!")
    else:
        # Contenitore Top 3 (Opzionale, ma utile per riassunto veloce)
        all_best_bets = [] 
        
        results_container = st.container()
        progress = st.progress(0)
        status = st.empty()
        api_map = {'I1':'soccer_italy_serie_a','E0':'soccer_epl','SP1':'soccer_spain_la_liga','D1':'soccer_germany_bundesliga','F1':'soccer_france_ligue_one'}
        
        step = 0
        for code, name in LEGHE.items():
            status.text(f"Analisi: {name}...")
            df_teams, df_matches = scarica_dati(code)
            
            if df_teams is not None:
                matches = get_live_matches(api_key_input, api_map[code])
                if matches:
                    for m in matches:
                        if 'home_team' not in m: continue
                        h, a = m['home_team'], m['away_team']
                        h_team = TEAM_MAPPING.get(h, h); a_team = TEAM_MAPPING.get(a, a)
                        
                        q1_b, qX_b, q2_b = 0,0,0
                        for b in m['bookmakers']:
                            for mk in b['markets']:
                                if mk['key'] == 'h2h':
                                    for o in mk['outcomes']:
                                        if o['name'] == h: q1_b = o['price']
                                        elif o['name'] == 'Draw': qX_b = o['price']
                                        elif o['name'] == a: q2_b = o['price']
                        if q1_b == 0: continue
                        
                        stats = get_full_stats(h_team, a_team, df_teams, df_matches)
                        if not stats: continue
                        
                        q1_m, qX_m, q2_m, lam_h, lam_a = calcola_1x2_lambda(stats['Shots'][0], stats['Shots'][1])
                        roi_1 = ((1/q1_m)*q1_b)-1; roi_X = ((1/qX_m)*qX_b)-1; roi_2 = ((1/q2_m)*q2_b)-1
                        
                        odds_dict = {'1': q1_b, 'X': qX_b, '2': q2_b}
                        roi_dict = {'1': roi_1, 'X': roi_X, '2': roi_2}
                        
                        # Generazione HTML Terminale
                        html_block = generate_complete_terminal(h_team, a_team, stats, lam_h, lam_a, odds_dict, roi_dict)
                        
                        with results_container:
                            st.markdown(html_block, unsafe_allow_html=True)
                                
            step += 1
            progress.progress(step / len(LEGHE))
            
        status.empty()
        st.success("Scansione Completata.")
