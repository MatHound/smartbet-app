import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from datetime import datetime

# Configurazione Pagina
st.set_page_config(page_title="SmartBet AI Mobile", page_icon="⚽", layout="wide")

# ==============================================================================
# 1. SETUP E INPUT (SIDEBAR)
# ==============================================================================
st.sidebar.title("⚙️ Configurazione")

# Input API Key (Sicurezza)
api_key_input = st.sidebar.text_input("Inserisci API Key (The-Odds-API)", type="password")

# Input Bankroll
bankroll_input = st.sidebar.number_input("Il tuo Bankroll (€)", min_value=10.0, value=26.50, step=0.5)

# Stagione
STAGIONE = "2526"
REGION = 'eu'
MARKET = 'h2h'

LEGHE = {
    'I1': '🇮🇹 SERIE A',
    'E0': '🇬🇧 PREMIER LEAGUE',
    'SP1': '🇪🇸 LA LIGA',
    'D1': '🇩🇪 BUNDESLIGA',
    'F1': '🇫🇷 LIGUE 1'
}

# Mapping Nomi
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

# ==============================================================================
# 2. FUNZIONI LOGICHE (CACHED)
# ==============================================================================
@st.cache_data(ttl=3600) # Cache per 1 ora per non scaricare sempre
def scarica_dati(codice_lega):
    url = f"https://www.football-data.co.uk/mmz4281/{STAGIONE}/{codice_lega}.csv"
    try:
        df = pd.read_csv(url)
    except:
        return None, None

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date')
    df['HomeTeam'] = df['HomeTeam'].str.strip()
    df['AwayTeam'] = df['AwayTeam'].str.strip()
    
    # Creazione dataset forza squadre
    df_temp = pd.concat([
        df[['Date','HomeTeam','AST']].rename(columns={'HomeTeam':'Team','AST':'Conceded'}),
        df[['Date','AwayTeam','HST']].rename(columns={'AwayTeam':'Team','HST':'Conceded'})
    ]).sort_values(['Team','Date'])
    
    defense_strength = df_temp.groupby('Team')['Conceded'].mean().to_dict()
    league_avg_conceded = df_temp['Conceded'].mean()
    
    def get_difficulty(opponent):
        opp_val = defense_strength.get(opponent, league_avg_conceded)
        return league_avg_conceded / opp_val if opp_val > 0 else 1.0

    # Metriche Casa
    cols_h = {'HST':'ShotsFor','AST':'ShotsAg','HC':'CornFor','AC':'CornAg','HF':'FoulsFor','AF':'FoulsAg','HY':'CardsFor','AY':'CardsAg'}
    home = df[['Date','HomeTeam', 'AwayTeam'] + list(cols_h.keys())].rename(columns={'HomeTeam':'Team', 'AwayTeam':'Opponent', **cols_h})
    home['IsHome'] = 1
    
    # Metriche Ospite
    cols_a = {'AST':'ShotsFor','HST':'ShotsAg','AC':'CornFor','HC':'CornAg','AF':'FoulsFor','HF':'FoulsAg','AY':'CardsFor','HY':'CardsAg'}
    away = df[['Date','AwayTeam', 'HomeTeam'] + list(cols_a.keys())].rename(columns={'AwayTeam':'Team', 'HomeTeam':'Opponent', **cols_a})
    away['IsHome'] = 0
    
    df_teams = pd.concat([home, away]).sort_values(['Team', 'Date'])
    df_teams['Diff_Factor'] = df_teams['Opponent'].map(lambda x: get_difficulty(x))
    
    # Calcolo Weighted Stats
    metrics = ['ShotsFor','ShotsAg','CornFor','CornAg', 'FoulsFor','FoulsAg','CardsFor','CardsAg']
    for m in metrics:
        if m in ['ShotsFor','ShotsAg','CornFor','CornAg']: # Applica diff factor solo alle tecniche
             df_teams[f'Adj_{m}'] = df_teams[m] * df_teams['Diff_Factor']
        else:
             df_teams[f'Adj_{m}'] = df_teams[m]

        form = df_teams.groupby('Team')[f'Adj_{m}'].transform(lambda x: x.shift(1).rolling(5).mean())
        season = df_teams.groupby('Team')[f'Adj_{m}'].transform(lambda x: x.shift(1).expanding().mean())
        df_teams[f'W_{m}'] = (form * 0.60) + (season * 0.40)

    return df_teams.dropna(), df

def get_live_matches(api_key, sport_key):
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions={REGION}&markets={MARKET}'
    try: 
        resp = requests.get(url)
        return resp.json()
    except: return []

def calcola_kelly(quota_reale, quota_book):
    if quota_book <= 1.01 or quota_reale <= 1.01: return 0.0
    b = quota_book - 1 
    p = 1 / quota_reale
    q = 1 - p
    kelly = (b * p - q) / b
    return round((kelly / 4) * 100, 2) if kelly > 0 else 0.0

def get_expected_stats(home, away, df_teams, df_matches):
    try:
        stats_h = df_teams[df_teams['Team'] == home].iloc[-1]
        stats_a = df_teams[df_teams['Team'] == away].iloc[-1]
    except: return None 

    results = {}
    config = [('Shots','HST','AST'), ('Corn','HC','AC'), ('Fouls','HF','AF'), ('Cards','HY','AY')]
    for name, ch, ca in config:
        avg_L_h = df_matches[ch].mean()
        avg_L_a = df_matches[ca].mean()
        
        att_h = stats_h[f'W_{name}For'] / avg_L_h
        def_a = stats_a[f'W_{name}Ag'] / avg_L_h
        exp_h = att_h * def_a * avg_L_h
        
        att_a = stats_a[f'W_{name}For'] / avg_L_a
        def_h = stats_h[f'W_{name}Ag'] / avg_L_a
        exp_a = att_a * def_h * avg_L_a
        results[name] = (exp_h, exp_a)
    return results

def calcola_1x2_lambda(exp_shots_h, exp_shots_a):
    lam_h, lam_a = exp_shots_h * 0.30, exp_shots_a * 0.30
    mat = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            mat[i,j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
    if lam_h < 1.3 and lam_a < 1.3: mat[0,0]*=1.15; mat[1,1]*=1.08 
    p1, pX, p2 = np.sum(np.tril(mat,-1)), np.trace(mat), np.sum(np.triu(mat,1))
    return (1/p1 if p1>0 else 99), (1/pX if pX>0 else 99), (1/p2 if p2>0 else 99), lam_h, lam_a

# Funzione Selettore Prop Migliore
def get_best_market_option(home_exp, away_exp, tot_exp, type_label, bankroll):
    if type_label == 'CORN':
        ranges_indiv = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        ranges_tot = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    elif type_label == 'FALLI':
        ranges_indiv = [8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
        ranges_tot = [21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5]
    elif type_label == 'GOL':
        ranges_indiv = [0.5, 1.5, 2.5]
        ranges_tot = [1.5, 2.5, 3.5]
    
    candidates = []
    
    def evaluate(label, exp, lines):
        for l in lines:
            prob = poisson.sf(int(l), exp)
            if prob > 0.70:
                quota = 1/prob if prob > 0 else 99
                candidates.append({
                    'desc': f"{label} Ov {l}",
                    'prob': prob,
                    'quota': quota
                })

    evaluate("CASA", home_exp, ranges_indiv)
    evaluate("OSPITE", away_exp, ranges_indiv)
    evaluate("TOT", tot_exp, ranges_tot)
    
    if not candidates:
        return None, None, None, None
    
    best = sorted(candidates, key=lambda x: x['prob'], reverse=True)[0]
    
    prob_perc = best['prob'] * 100
    icon = "💎" if best['prob'] > 0.80 else "🔥" if best['prob'] > 0.70 else ""
    
    stake = round(bankroll * 0.05, 2)
    if best['prob'] > 0.80: stake = round(bankroll * 0.10, 2)
    if stake < 0.5: stake = 0.5
    
    return f"{icon} {best['desc']}", best['quota'], stake, prob_perc

# ==============================================================================
# 3. INTERFACCIA STREAMLIT
# ==============================================================================
st.title("📱 SmartBet AI: Mobile Dashboard")
st.markdown("### Il tuo Assistente Quantitativo Tascabile")

if st.button("🚀 AVVIA ANALISI", type="primary"):
    if not api_key_input:
        st.error("Inserisci prima la tua API Key nella barra laterale!")
    else:
        results_list = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        api_sports = {
            'I1': 'soccer_italy_serie_a', 'E0': 'soccer_epl',
            'SP1': 'soccer_spain_la_liga', 'D1': 'soccer_germany_bundesliga',
            'F1': 'soccer_france_ligue_one'
        }
        
        step = 0
        total_steps = len(LEGHE)

        for code, name in LEGHE.items():
            status_text.text(f"Analisi in corso: {name}...")
            df_teams, df_matches = scarica_dati(code)
            
            if df_teams is None:
                st.warning(f"Dati storici non trovati per {name}")
                continue
                
            matches = get_live_matches(api_key_input, api_sports[code])
            
            if not matches:
                continue
                
            for m in matches:
                if 'home_team' not in m: continue
                h_api, a_api = m['home_team'], m['away_team']
                h_team = TEAM_MAPPING.get(h_api, h_api)
                a_team = TEAM_MAPPING.get(a_api, a_api)
                
                # Quote Bookmaker
                q1_b, qX_b, q2_b = 0,0,0
                for b in m['bookmakers']:
                    for mk in b['markets']:
                        if mk['key'] == 'h2h':
                            for o in mk['outcomes']:
                                if o['name'] == h_api: q1_b = o['price']
                                elif o['name'] == 'Draw': qX_b = o['price']
                                elif o['name'] == a_api: q2_b = o['price']
                    if q1_b > 0: break
                
                if q1_b == 0: continue
                
                stats = get_expected_stats(h_team, a_team, df_teams, df_matches)
                if not stats: continue
                
                # 1X2 Logic
                q1_m, qX_m, q2_m, lam_h, lam_a = calcola_1x2_lambda(stats['Shots'][0], stats['Shots'][1])
                roi_1 = ((1/q1_m)*q1_b)-1
                roi_X = ((1/qX_m)*qX_b)-1
                roi_2 = ((1/q2_m)*q2_b)-1
                
                best_roi = max(roi_1, roi_X, roi_2)
                best_segno = "1" if roi_1 == best_roi else "X" if roi_X == best_roi else "2"
                best_quota = q1_b if roi_1 == best_roi else qX_b if roi_X == best_roi else q2_b
                
                stake_1x2 = 0
                if best_roi > 0.05:
                    stake_1x2 = round(bankroll_input * 0.05, 2) # Base stake
                
                # Prop Logic
                tot_corn_exp = stats['Corn'][0] + stats['Corn'][1]
                tot_fouls_exp = stats['Fouls'][0] + stats['Fouls'][1]
                tot_gol_exp = lam_h + lam_a
                
                bet_c, q_c, s_c, p_c = get_best_market_option(stats['Corn'][0], stats['Corn'][1], tot_corn_exp, 'CORN', bankroll_input)
                bet_f, q_f, s_f, p_f = get_best_market_option(stats['Fouls'][0], stats['Fouls'][1], tot_fouls_exp, 'FALLI', bankroll_input)
                bet_g, q_g, s_g, p_g = get_best_market_option(lam_h, lam_a, tot_gol_exp, 'GOL', bankroll_input)
                
                # Salva Riga se c'è valore
                if best_roi > 0.05 or bet_c or bet_f or bet_g:
                    results_list.append({
                        'LEGA': name,
                        'MATCH': f"{h_team} - {a_team}",
                        '1X2_BET': best_segno if best_roi > 0.05 else "-",
                        '1X2_ROI': f"{best_roi*100:.1f}%" if best_roi > 0.05 else "-",
                        '1X2_STAKE': f"€ {stake_1x2}" if best_roi > 0.05 else "-",
                        
                        'CORN_BET': bet_c if bet_c else "-",
                        'CORN_Q': f"{q_c:.2f}" if q_c else "-",
                        'CORN_STAKE': f"€ {s_c}" if s_c else "-",
                        
                        'FALLI_BET': bet_f if bet_f else "-",
                        'FALLI_Q': f"{q_f:.2f}" if q_f else "-",
                        'FALLI_STAKE': f"€ {s_f}" if s_f else "-",
                        
                        'GOL_BET': bet_g if bet_g else "-",
                        'GOL_Q': f"{q_g:.2f}" if q_g else "-",
                        'GOL_STAKE': f"€ {s_g}" if s_g else "-",
                        'SORT_KEY': best_roi # Chiave nascosta per ordinamento
                    })
            
            step += 1
            progress_bar.progress(step / total_steps)

        status_text.text("Analisi Completata!")
        
        if results_list:
            df = pd.DataFrame(results_list)
            df = df.sort_values(by='SORT_KEY', ascending=False).drop(columns=['SORT_KEY'])
            
            st.success(f"Trovate {len(df)} Opportunità di Betting!")
            
            # Mostra Tabella Interattiva
            st.dataframe(
                df,
                column_config={
                    "MATCH": st.column_config.TextColumn("Partita", width="medium"),
                    "1X2_BET": st.column_config.TextColumn("1X2", width="small"),
                    "CORN_BET": st.column_config.TextColumn("Corner", width="medium"),
                    "FALLI_BET": st.column_config.TextColumn("Falli", width="medium"),
                    "GOL_BET": st.column_config.TextColumn("Gol", width="medium"),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("Nessuna opportunità trovata oggi con i parametri attuali.")
