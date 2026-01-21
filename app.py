import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from datetime import datetime, timedelta

# ==============================================================================
# 1. CONFIGURAZIONE E COSTANTI
# ==============================================================================
st.set_page_config(page_title="SmartBet Sniper", page_icon="🎯", layout="centered")

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
    .term-green { color: #00FF00; font-weight: bold; } /* Verde Matrix */
    .term-dim { color: #555555; } /* Grigio scuro per roba rischiosa */
    .term-val { color: #FF00FF; font-weight: bold; }
    .streamlit-expanderHeader { font-weight: bold; background-color: #f0f2f6; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# DATABASE LEGHE
ALL_LEAGUES = {
    'I1': '🇮🇹 ITA - Serie A', 'E0': '🇬🇧 ENG - Premier League', 'SP1': '🇪🇸 ESP - La Liga',
    'D1': '🇩🇪 GER - Bundesliga', 'F1': '🇫🇷 FRA - Ligue 1', 'I2': '🇮🇹 ITA - Serie B',
    'E1': '🇬🇧 ENG - Championship', 'N1': '🇳🇱 NED - Eredivisie', 'P1': '🇵🇹 POR - Primeira Liga',
    'B1': '🇧🇪 BEL - Pro League', 'T1': '🇹🇷 TUR - Super Lig'
}

API_MAPPING = {
    'I1': 'soccer_italy_serie_a', 'I2': 'soccer_italy_serie_b',
    'E0': 'soccer_epl', 'E1': 'soccer_efl_champ',
    'SP1': 'soccer_spain_la_liga', 'D1': 'soccer_germany_bundesliga',
    'F1': 'soccer_france_ligue_one', 'N1': 'soccer_netherlands_eredivisie',
    'P1': 'soccer_portugal_primeira_liga', 'B1': 'soccer_belgium_pro_league',
    'T1': 'soccer_turkey_super_league'
}

# --- MEGA MAPPING ---
TEAM_MAPPING = {
    # ITALIA
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
    
    # INGHILTERRA
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
    
    # SPAGNA
    'Atletico Madrid': 'Ath Madrid', 'Athletic Bilbao': 'Ath Bilbao', 'Real Betis': 'Betis',
    'Real Sociedad': 'Sociedad', 'Rayo Vallecano': 'Vallecano', 'Celta Vigo': 'Celta', 
    'Alavés': 'Alaves', 'Cadiz CF': 'Cadiz', 'UD Las Palmas': 'Las Palmas', 'RCD Espanyol': 'Espanyol',
    'Real Valladolid': 'Valladolid', 'Leganés': 'Leganes', 'Girona FC': 'Girona',
    
    # GERMANIA
    'Bayern Munich': 'Bayern Munich', 'Bayer Leverkusen': 'Leverkusen', 'Borussia Dortmund': 'Dortmund',
    'Borussia Monchengladbach': "M'gladbach", 'Eintracht Frankfurt': 'Ein Frankfurt', 
    'SC Freiburg': 'Freiburg', '1. FC Köln': 'FC Koln', 'Mainz 05': 'Mainz', 'VfL Bochum': 'Bochum',
    'VfB Stuttgart': 'Stuttgart', 'FC St. Pauli': 'St Pauli', 'Holstein Kiel': 'Holstein Kiel',
    'TSG Hoffenheim': 'Hoffenheim', 'Werder Bremen': 'Werder Bremen', 'Augsburg': 'Augsburg',
    
    # FRANCIA
    'Paris Saint Germain': 'Paris SG', 'Marseille': 'Marseille', 'Saint-Etienne': 'St Etienne',
    'Clermont Foot': 'Clermont', 'Le Havre': 'Le Havre', 'RC Lens': 'Lens', 'AS Monaco': 'Monaco',
    'Lille OSC': 'Lille', 'Olympique Lyonnais': 'Lyon',
    
    # OLANDA
    'PSV Eindhoven': 'PSV Eindhoven', 'Feyenoord Rotterdam': 'Feyenoord', 'Ajax Amsterdam': 'Ajax',
    'AZ Alkmaar': 'AZ Alkmaar', 'FC Twente': 'Twente', 'FC Utrecht': 'Utrecht',
    'Sparta Rotterdam': 'Sparta Rotterdam', 'NEC Nijmegen': 'Nijmegen', 'Go Ahead Eagles': 'Go Ahead Eagles',
    'Fortuna Sittard': 'For Sittard', 'PEC Zwolle': 'Zwolle', 'Almere City': 'Almere City',
    'RKC Waalwijk': 'Waalwijk', 'SC Heerenveen': 'Heerenveen', 'Heracles Almelo': 'Heracles',
    
    # PORTOGALLO
    'Sporting CP': 'Sp Lisbon', 'Sporting Lisbon': 'Sp Lisbon', 
    'Benfica': 'Benfica', 'FC Porto': 'Porto', 'Sporting Braga': 'Braga', 'Vitoria Guimaraes': 'Guimaraes',
    'Boavista FC': 'Boavista', 'Estoril Praia': 'Estoril', 'Casa Pia AC': 'Casa Pia',
    'Farense': 'Farense', 'Arouca': 'Arouca', 'Gil Vicente': 'Gil Vicente',
    
    # TURCHIA
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
    url = f"https://www.football-data.co.uk/mmz4281/{STAGIONE}/{codice_lega}.csv"
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date')
        df['HomeTeam'] = df['HomeTeam'].str.strip(); df['AwayTeam'] = df['AwayTeam'].str.strip()
        
        needed = ['Date','HomeTeam','AwayTeam','FTHG','FTAG']
        if not all(col in df.columns for col in needed): return None, None
        
        for col in ['HST','AST','HC','AC','HF','AF','HY','AY']:
            if col not in df.columns: df[col] = 0
            
        df_temp = pd.concat([
            df[['Date','HomeTeam','FTAG']].rename(columns={'HomeTeam':'Team','FTAG':'Conceded'}),
            df[['Date','AwayTeam','FTHG']].rename(columns={'AwayTeam':'Team','FTHG':'Conceded'})
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
    lam_h = exp_shots_h * 0.30 if exp_shots_h > 0 else 1.0
    lam_a = exp_shots_a * 0.30 if exp_shots_a > 0 else 0.8
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
    p_h = np.sum(np.tril(joint, -1)); p_a = np.sum(np.triu(joint, 1))
    return p_h, p_a

def get_full_stats(home, away, df_teams, df_matches):
    try:
        s_h = df_teams[df_teams['Team'] == home].iloc[-1]
        s_a = df_teams[df_teams['Team'] == away].iloc[-1]
    except: return None
    res = {}
    config = [('Shots','HST','AST'), ('Corn','HC','AC'), ('Fouls','HF','AF'), ('Cards','HY','AY')]
    for name, ch, ca in config:
        avg_L_h = df_matches[ch].mean() if df_matches[ch].mean() > 0 else 1
        avg_L_a = df_matches[ca].mean() if df_matches[ca].mean() > 0 else 1
        att_h = s_h[f'W_{name}For'] / avg_L_h; def_a = s_a[f'W_{name}Ag'] / avg_L_h; exp_h = att_h * def_a * avg_L_h
        att_a = s_a[f'W_{name}For'] / avg_L_a; def_h = s_h[f'W_{name}Ag'] / avg_L_a; exp_a = att_a * def_h * avg_L_a
        res[name] = (exp_h, exp_a)
    return res

def generate_complete_terminal(h_team, a_team, stats, lam_h, lam_a, odds_1x2, roi_1x2, min_prob):
    html = f"""<div class='terminal-box'>"""
    
    # 1X2 (Solo Informativo, spesso rischioso)
    html += f"<span class='term-section'>[ 1X2 ANALYSIS ]</span>\n"
    html += f"{'SEGNO':<6} | {'MY QUOTA':<10} | {'BOOKIE':<8} | {'VALUE'}\n"
    html += "-"*45 + "\n"
    segni = [('1', roi_1x2['1'], odds_1x2['1']), ('X', roi_1x2['X'], odds_1x2['X']), ('2', roi_1x2['2'], odds_1x2['2'])]
    for segno, roi, book_q in segni:
        my_q = book_q / (roi + 1) if (roi+1) > 0 else 99.0
        val_str = f"{roi*100:+.0f}%"
        # Valore Verde SOLO se la probabilità è alta (Q bassa) E c'è valore
        implied_prob = 1/my_q if my_q > 0 else 0
        
        if roi >= 0.15 and book_q <= 4.0 and implied_prob >= min_prob: # FILTRO CECCHINO SU 1X2
            val_str = f"<span class='term-val'>{val_str} (TOP)</span>"
        elif roi > 0 and implied_prob >= min_prob:
            val_str = f"<span class='term-green'>{val_str}</span>"
        else:
            val_str = f"<span class='term-dim'>{val_str}</span>"
            
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
                
                # FILTRO CECCHINO (VISIBILITA')
                if p >= min_prob: 
                    rows_html += f"<span class='term-green'>{row_str}</span>\n"
                elif p >= min_prob - 0.10: # Mostra ma grigio se vicino
                    rows_html += f"<span class='term-dim'>{row_str}</span>\n"
                # Se è troppo basso non lo aggiungo proprio per pulizia (o lo lascio dim)
            return rows_html
            
        html += add_rows("CASA", r_h, exp_h)
        html += add_rows("OSP", r_a, exp_a)
        html += add_rows("TOT", r_tot, exp_h+exp_a)

    html += "</div>"
    return html

# ==============================================================================
# 3. INTERFACCIA UTENTE
# ==============================================================================

with st.sidebar:
    st.header("🎯 Sniper Setup")
    api_key_input = st.text_input("API Key", type="password")
    bankroll_input = st.number_input("Bankroll (€)", min_value=10.0, value=26.50, step=0.5)
    
    st.divider()
    st.markdown("### 🛡️ Livello Sicurezza")
    # SLIDER PROBABILITA' MINIMA
    min_prob_val = st.slider("Probabilità Minima (Cecchino)", 0.50, 0.90, 0.65, step=0.05)
    st.caption(f"Mostro in VERDE solo giocate con prob > {min_prob_val*100:.0f}%")
    
    st.divider()
    st.markdown("### 🏆 Campionati")
    selected_leagues_keys = st.multiselect(
        "Scegli le leghe:",
        options=list(ALL_LEAGUES.keys()),
        format_func=lambda x: ALL_LEAGUES[x],
        default=['I1', 'E0', 'SP1', 'D1', 'F1']
    )
    
    show_mapping_errors = st.checkbox("🛠️ Mostra Errori Mapping", value=False)

st.title("🎯 SmartBet Sniper Terminal")
st.caption(f"Modalità Sicura Attiva (> {min_prob_val*100:.0f}%)")

start_analisys = st.button("🚀 CERCA VALUE BETS", type="primary", use_container_width=True)

# MAIN LOOP
if start_analisys:
    if not api_key_input:
        st.error("Inserisci l'API Key!")
    elif not selected_leagues_keys:
        st.error("Seleziona almeno un campionato!")
    else:
        results_by_league = {ALL_LEAGUES[k]: [] for k in selected_leagues_keys}
        global_calendar_data = [] 
        missing_teams_log = [] 
        
        progress = st.progress(0)
        status = st.empty()
        
        step = 0
        total_steps = len(selected_leagues_keys)
        
        for code in selected_leagues_keys:
            league_name = ALL_LEAGUES[code]
            status.text(f"Analisi: {league_name}...")
            
            df_teams, df_matches = scarica_dati(code)
            
            if df_teams is not None:
                matches = get_live_matches(api_key_input, API_MAPPING.get(code, ''))
                available_teams = set(df_teams['Team'].unique())
                
                if matches:
                    for m in matches:
                        if 'home_team' not in m: continue
                        
                        h_raw, a_raw = m['home_team'], m['away_team']
                        h_team = TEAM_MAPPING.get(h_raw, h_raw)
                        a_team = TEAM_MAPPING.get(a_raw, a_raw)
                        
                        if h_team not in available_teams:
                            missing_teams_log.append(f"LEGA {code}: '{h_raw}' (API) -> Non trovato")
                            continue 
                        if a_team not in available_teams:
                            missing_teams_log.append(f"LEGA {code}: '{a_raw}' (API) -> Non trovato")
                            continue 
                        
                        raw_date_obj, fmt_date_str = parse_date(m.get('commence_time', ''))
                        
                        q1_b, qX_b, q2_b = 0,0,0
                        for b in m['bookmakers']:
                            for mk in b['markets']:
                                if mk['key'] == 'h2h':
                                    for o in mk['outcomes']:
                                        if o['name'] == h_raw: q1_b = o['price']
                                        elif o['name'] == 'Draw': qX_b = o['price']
                                        elif o['name'] == a_raw: q2_b = o['price']
                        if q1_b == 0: continue
                        
                        stats = get_full_stats(h_team, a_team, df_teams, df_matches)
                        if not stats: continue
                        
                        q1_m, qX_m, q2_m, lam_h, lam_a = calcola_1x2_lambda(stats['Shots'][0], stats['Shots'][1])
                        roi_1 = ((1/q1_m)*q1_b)-1; roi_X = ((1/qX_m)*qX_b)-1; roi_2 = ((1/q2_m)*q2_b)-1
                        
                        # PASSO IL PARAMETRO MIN_PROB PER IL FILTRO GRAFICO
                        html_block = generate_complete_terminal(
                            h_team, a_team, stats, lam_h, lam_a, 
                            {'1':q1_b,'X':qX_b,'2':q2_b}, 
                            {'1':roi_1,'X':roi_X,'2':roi_2},
                            min_prob_val
                        )
                        
                        item = {'label': f"{fmt_date_str} | {h_team} vs {a_team}", 'html': html_block}
                        results_by_league[league_name].append(item)
                        global_calendar_data.append({'date': raw_date_obj, 'label': f"[{code}] {h_team} vs {a_team}", 'html': html_block})
                                
            step += 1
            progress.progress(step / total_steps)
            
        status.empty()
        
        if show_mapping_errors and missing_teams_log:
            st.warning(f"⚠️ TROVATI {len(missing_teams_log)} ERRORI DI MAPPING")
            unique_missing = list(set(missing_teams_log))
            for err in unique_missing: st.error(err)
        
        st.success("Analisi Completata.")
        
        main_tab1, main_tab2 = st.tabs(["🏆 ANALISI LEGHE", "📅 CALENDARIO GLOBALE"])
        
        with main_tab1:
            league_tabs = st.tabs([ALL_LEAGUES[k] for k in selected_leagues_keys])
            for i, code in enumerate(selected_leagues_keys):
                name = ALL_LEAGUES[code]
                with league_tabs[i]:
                    matches = results_by_league[name]
                    if not matches: st.write("Nessuna partita rilevata.")
                    else:
                        for match in matches:
                            with st.expander(match['label']): st.markdown(match['html'], unsafe_allow_html=True)
                                
        with main_tab2:
            st.markdown("#### Seleziona un giorno")
            if global_calendar_data:
                available_dates = sorted(list(set([d['date'] for d in global_calendar_data])))
                if available_dates:
                    min_d = available_dates[0]
                    selected_date = st.date_input("Data:", value=min_d, min_value=min_d)
                    filtered = [m for m in global_calendar_data if m['date'] == selected_date]
                    
                    if filtered:
                        st.info(f"{len(filtered)} partite il {selected_date.strftime('%d/%m')}")
                        for match in filtered:
                            with st.expander(match['label']): st.markdown(match['html'], unsafe_allow_html=True)
                    else: st.warning("Nessuna partita in questa data.")
            else: st.write("Nessun dato disponibile.")
