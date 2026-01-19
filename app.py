import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

# Configurazione Pagina
st.set_page_config(page_title="SmartBet v25.4", page_icon="⚽", layout="centered")

# CSS UI
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #00cc00; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #e6ffe6; border-bottom: 2px solid #00cc00; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Setup")
    api_key_input = st.text_input("API Key", type="password")
    bankroll_input = st.number_input("Bankroll (€)", min_value=10.0, value=26.50, step=0.5)
    st.success("Versione: v25.4 (Clean)")

# TITOLO CON VERSIONE VISIBILE
st.title("⚽ SmartBet AI (v25.4)")
st.caption(f"Bankroll Attuale: €{bankroll_input:.2f}")

start_analisys = st.button("🚀 CERCA VALUE BETS", type="primary", use_container_width=True)

# COSTANTI
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
    # Calcolo LAMBDA (Gol Attesi)
    lam_h = exp_shots_h * 0.30
    lam_a = exp_shots_a * 0.30
    
    # Matrice Poisson
    mat = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            mat[i,j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            
    if lam_h < 1.3 and lam_a < 1.3: mat[0,0]*=1.15; mat[1,1]*=1.08 
    
    p1 = np.sum(np.tril(mat,-1))
    pX = np.trace(mat)
    p2 = np.sum(np.triu(mat,1))
    
    # IMPORTANTISSIMO: Restituisce 5 valori
    # 1: Quota Casa
    # 2: Quota X
    # 3: Quota Ospite
    # 4: Gol Attesi Casa (lam_h)
    # 5: Gol Attesi Ospite (lam_a)
    return (1/p1 if p1>0 else 99), (1/pX if pX>0 else 99), (1/p2 if p2>0 else 99), lam_h, lam_a

def get_full_stats(home, away, df_teams, df_matches):
    try:
        s_h = df_teams[df_teams['Team'] == home].iloc[-1]
        s_a = df_teams[df_teams['Team'] == away].iloc[-1]
    except: return None

    res = {}
    config = [('Shots','HST','AST'), ('Corn','HC','AC'), ('Fouls','HF','AF'), ('Cards','HY','AY')]
    
    for name, ch, ca in config:
        avg_L_h = df_matches[ch].mean()
        avg_L_a = df_matches[ca].mean()
        
        att_h = s_h[f'W_{name}For'] / avg_L_h; def_a = s_a[f'W_{name}Ag'] / avg_L_h
        exp_h = att_h * def_a * avg_L_h
        
        att_a = s_a[f'W_{name}For'] / avg_L_a; def_h = s_h[f'W_{name}Ag'] / avg_L_a
        exp_a = att_a * def_h * avg_L_a
        
        res[name] = (exp_h, exp_a)
    return res

def get_best_prop(home_exp, away_exp, label, bankroll):
    if label == 'CORN':
        ranges_indiv = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        ranges_tot = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    elif label == 'FALLI':
        ranges_indiv = [8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
        ranges_tot = [21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5]
    elif label == 'GOL':
        ranges_indiv = [0.5, 1.5, 2.5]
        ranges_tot = [1.5, 2.5, 3.5]
        
    tot_exp = home_exp + away_exp
    opts = []
    
    def check(lbl, exp, lines):
        for l in lines:
            p = poisson.sf(int(l), exp)
            if p > 0.70: 
                q = 1/p if p > 0 else 1.01
                opts.append({'desc':f"{lbl} Ov {l}", 'prob':p, 'q':q})
            
    check("CASA", home_exp, ranges_indiv)
    check("OSP", away_exp, ranges_indiv)
    check("TOT", tot_exp, ranges_tot)
    
    if not opts: return None
    
    # VALUE SORTING (Quota più alta tra le prob > 70%)
    best = sorted(opts, key=lambda x: x['q'], reverse=True)[0]
    
    stake = round(bankroll * 0.05, 2)
    if best['prob'] > 0.80: stake = round(bankroll * 0.10, 2)
    if stake < 0.5: stake = 0.5
    
    return {'desc': best['desc'], 'prob': best['prob'], 'q': best['q'], 'stake': stake}

# MAIN LOOP
if start_analisys:
    if not api_key_input:
        st.error("Inserisci l'API Key nel menu laterale!")
    else:
        results_by_league = {name: [] for code, name in LEGHE.items()}
        all_bets = []
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
                        
                        q_book = 0
                        for b in m['bookmakers']:
                            for mk in b['markets']:
                                if mk['key'] == 'h2h':
                                    for o in mk['outcomes']:
                                        if o['name'] == h: q_book = o['price']
                            if q_book > 0: break
                        if q_book == 0: continue
                        
                        stats = get_full_stats(h_team, a_team, df_teams, df_matches)
                        if not stats: continue
                        
                        # --- FIX CRITICO: UNPACKING 5 VALORI ---
                        _, _, _, lam_h, lam_a = calcola_1x2_lambda(stats['Shots'][0], stats['Shots'][1])
                        
                        p_corn = get_best_prop(stats['Corn'][0], stats['Corn'][1], 'CORN', bankroll_input)
                        p_foul = get_best_prop(stats['Fouls'][0], stats['Fouls'][1], 'FALLI', bankroll_input)
                        p_gol = get_best_prop(lam_h, lam_a, 'GOL', bankroll_input)
                        
                        match_data = {
                            'match': f"{h_team} vs {a_team}", 
                            'props': [],
                            # DEBUG DATA (Per verificare i numeri)
                            'debug_gol': f"Exp Gol: {lam_h:.2f} vs {lam_a:.2f}"
                        }
                        if p_corn: match_data['props'].append(p_corn)
                        if p_foul: match_data['props'].append(p_foul)
                        if p_gol: match_data['props'].append(p_gol)
                        
                        if match_data['props']:
                            results_by_league[name].append(match_data)
                            for p in match_data['props']:
                                p['match'] = match_data['match']
                                all_bets.append(p)
                                
            step += 1
            progress.progress(step / len(LEGHE))
            
        status.empty()
        
        if all_bets:
            st.markdown("### 🔥 Top 3 Migliori Giocate")
            top_bets = sorted(all_bets, key=lambda x: x['prob'], reverse=True)[:3]
            cols = st.columns(3)
            for i, bet in enumerate(top_bets):
                with cols[i]:
                    st.info(f"**{bet['match']}**\n\n{bet['desc']}")
                    st.metric("Quota Reale", f"{bet['q']:.2f}", f"Prob: {bet['prob']*100:.0f}%")
                    st.write(f"💶 **Punta:** €{bet['stake']:.2f}")
        
        st.divider()
        tabs = st.tabs(list(LEGHE.values()))
        
        for i, (code, name) in enumerate(LEGHE.items()):
            with tabs[i]:
                matches = results_by_league[name]
                if not matches:
                    st.write("Nessuna opportunità.")
                else:
                    for m in matches:
                        with st.container(border=True):
                            st.subheader(m['match'])
                            # Visualizza i Gol Attesi per Debug
                            st.caption(f"📊 Dati Tecnici: {m['debug_gol']}") 
                            
                            p_cols = st.columns(len(m['props'])) if m['props'] else [st.container()]
                            for idx, p in enumerate(m['props']):
                                with p_cols[idx]:
                                    st.markdown(f"**{p['desc']}**")
                                    conf_color = "#00cc00" if p['prob'] > 0.8 else "#ffcc00"
                                    st.progress(p['prob'], text=f"Confidenza: {p['prob']*100:.0f}%")
                                    c1, c2 = st.columns(2)
                                    c1.metric("Quota", f"{p['q']:.2f}")
                                    c2.metric("Stake", f"€{p['stake']}")
