import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

# Configurazione Pagina
st.set_page_config(page_title="SmartBet Hybrid", page_icon="⚽", layout="centered")

# CSS Custom
st.markdown("""
<style>
    div[data-testid="column"] { background-color: #f9f9f9; border-radius: 5px; padding: 10px; border: 1px solid #ddd; }
    h4 { margin-top: 0px; margin-bottom: 5px; font-size: 1rem; }
    .stMetric { text-align: center; }
    .value-box { background-color: #d4edda; border: 2px solid #28a745; padding: 10px; border-radius: 5px; text-align: center; color: #155724; font-weight: bold; }
    .neutral-box { background-color: #f8f9fa; border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center; color: #666; }
    .small-text { font-size: 0.85em; color: #555; }
    .prob-text { font-size: 0.8em; font-weight: bold; color: #333; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Setup")
    api_key_input = st.text_input("API Key", type="password")
    bankroll_input = st.number_input("Bankroll (€)", min_value=10.0, value=26.50, step=0.5)
    min_quota_filter = st.slider("Filtro Quota (Vista Smart)", 1.10, 1.50, 1.25, step=0.05)
    st.info(f"v32.0 - Vista Terminale Inclusa")

st.title("⚽ SmartBet AI Dashboard")
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
    if p_h > p_a: return "CASA", p_h
    else: return "OSP", p_a

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

# FUNZIONE PER GENERARE IL REPORT VECCHIO STILE (RAW)
def generate_terminal_report(h_team, a_team, stats, lam_h, lam_a):
    output = f"ANALISI RAW: {h_team} vs {a_team}\n"
    output += "="*40 + "\n"
    
    # Configurazione Range (uguale al vecchio script)
    configs = [
        ("CORNER", stats['Corn'][0], stats['Corn'][1], [3.5, 4.5, 5.5], [2.5, 3.5, 4.5], [8.5, 9.5, 10.5]),
        ("TIRI PORTA", stats['Shots'][0], stats['Shots'][1], [3.5, 4.5, 5.5], [2.5, 3.5, 4.5], [7.5, 8.5, 9.5]),
        ("FALLI", stats['Fouls'][0], stats['Fouls'][1], [10.5, 11.5, 12.5], [10.5, 11.5, 12.5], [21.5, 22.5, 23.5]),
        ("GOL (Est.)", lam_h, lam_a, [0.5, 1.5], [0.5, 1.5], [1.5, 2.5, 3.5])
    ]
    
    for label, exp_h, exp_a, r_h, r_a, r_tot in configs:
        output += f"\n>>> {label} (Att: {exp_h:.2f} - {exp_a:.2f} | Tot: {exp_h+exp_a:.2f})\n"
        output += f"{'LINEA':<12} | {'PROB %':<6} | {'QUOTA':<6}\n"
        output += "-"*35 + "\n"
        
        # Casa
        for l in r_h:
            p = poisson.sf(int(l), exp_h)
            output += f"{'CASA Ov '+str(l):<12} | {p*100:04.1f}% | {1/p if p>0 else 99:.2f}\n"
        
        # Ospite
        for l in r_a:
            p = poisson.sf(int(l), exp_a)
            output += f"{'OSP Ov '+str(l):<12} | {p*100:04.1f}% | {1/p if p>0 else 99:.2f}\n"
            
        # Totale
        for l in r_tot:
            p = poisson.sf(int(l), exp_h+exp_a)
            output += f"{'TOT Ov '+str(l):<12} | {p*100:04.1f}% | {1/p if p>0 else 99:.2f}\n"

    return output

def get_props_list(home_exp, away_exp, label, bankroll, min_quota):
    # Logica Smart (Filtrata)
    if label == 'CORN':
        ranges_indiv = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        ranges_tot = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    elif label == 'FALLI':
        ranges_indiv = [8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
        ranges_tot = [21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5]
    elif label == 'GOL':
        ranges_indiv = [0.5, 1.5, 2.5]
        ranges_tot = [1.5, 2.5, 3.5]
    elif label == 'SHOTS':
        ranges_indiv = [2.5, 3.5, 4.5, 5.5, 6.5]
        ranges_tot = [7.5, 8.5, 9.5, 10.5, 11.5]
    elif label == 'CARDS':
        ranges_indiv = [0.5, 1.5, 2.5]
        ranges_tot = [2.5, 3.5, 4.5, 5.5]
        
    tot_exp = home_exp + away_exp
    valid_opts = []
    
    def check(lbl, exp, lines):
        for l in lines:
            p = poisson.sf(int(l), exp)
            if p > 0.70: 
                q = 1/p if p > 0 else 1.01
                if q < min_quota: continue
                stake = round(bankroll * 0.05, 2)
                if p > 0.80: stake = round(bankroll * 0.10, 2)
                valid_opts.append({'type': label, 'desc': f"{lbl} Ov {l}", 'prob': p, 'q': q, 'stake': stake})
            
    check("CASA", home_exp, ranges_indiv)
    check("OSP", away_exp, ranges_indiv)
    check("TOT", tot_exp, ranges_tot)
    
    return sorted(valid_opts, key=lambda x: x['q'], reverse=True)[:2]

def custom_progress(prob):
    pct = int(prob * 100)
    if pct >= 90: color = "#006400"
    elif pct >= 80: color = "#00cc00"
    else: color = "#85e085"
    return f"""<div style="width:100%; background-color: #e0e0e0; border-radius: 5px; height: 8px; margin-top:5px; margin-bottom:2px;"><div style="width:{pct}%; background-color: {color}; height: 8px; border-radius: 5px;"></div></div>"""

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
                        
                        h2h_data = []
                        metrics_cfg = [("Tiri", 'Shots'), ("Corner", 'Corn'), ("Falli", 'Fouls'), ("Cards", 'Cards')]
                        for label, key in metrics_cfg:
                            fav, prob = calcola_h2h_favorito(stats[key][0], stats[key][1])
                            fav_team = "🏠" if fav == "CASA" else "✈️"
                            h2h_data.append({'metric': label, 'fav': fav_team, 'prob': prob})
                        
                        min_q = min_quota_filter
                        list_corn = get_props_list(stats['Corn'][0], stats['Corn'][1], 'CORN', bankroll_input, min_q)
                        list_foul = get_props_list(stats['Fouls'][0], stats['Fouls'][1], 'FALLI', bankroll_input, min_q)
                        list_gol = get_props_list(lam_h, lam_a, 'GOL', bankroll_input, min_q)
                        list_shots = get_props_list(stats['Shots'][0], stats['Shots'][1], 'SHOTS', bankroll_input, min_q)
                        list_cards = get_props_list(stats['Cards'][0], stats['Cards'][1], 'CARDS', bankroll_input, min_q)
                        
                        match_name = f"{h_team} vs {a_team}"
                        all_lists = [list_corn, list_foul, list_gol, list_shots, list_cards]
                        for l in all_lists:
                            for b in l:
                                b['match'] = match_name
                        
                        # Generazione Report Terminale
                        raw_report = generate_terminal_report(h_team, a_team, stats, lam_h, lam_a)

                        match_data = {
                            'match': match_name,
                            '1x2': {'1': {'q': q1_b, 'roi': roi_1}, 'X': {'q': qX_b, 'roi': roi_X}, '2': {'q': q2_b, 'roi': roi_2}},
                            'h2h': h2h_data,
                            'exp_corn': (stats['Corn'][0], stats['Corn'][1]),
                            'exp_foul': (stats['Fouls'][0], stats['Fouls'][1]),
                            'exp_gol': (lam_h, lam_a),
                            'exp_shots': (stats['Shots'][0], stats['Shots'][1]),
                            'exp_cards': (stats['Cards'][0], stats['Cards'][1]),
                            'corn_bets': list_corn, 'foul_bets': list_foul, 'gol_bets': list_gol, 
                            'shots_bets': list_shots, 'cards_bets': list_cards,
                            'raw_text': raw_report
                        }
                        
                        has_val_1x2 = max(roi_1, roi_X, roi_2) > 0.05
                        has_props = list_corn or list_foul or list_gol or list_shots or list_cards
                        
                        if has_val_1x2 or has_props:
                            results_by_league[name].append(match_data)
                            all_bets.extend(list_corn + list_foul + list_gol + list_shots + list_cards)
                                
            step += 1
            progress.progress(step / len(LEGHE))
            
        status.empty()
        
        if all_bets:
            st.markdown("### 🔥 Top 3 Value Picks")
            top_bets = sorted(all_bets, key=lambda x: x['prob'], reverse=True)[:3]
            cols = st.columns(3)
            for i, bet in enumerate(top_bets):
                icon = '🚩' if bet['type']=='CORN' else '🛑' if bet['type']=='FALLI' else '⚽'
                if bet['type'] == 'SHOTS': icon = '🎯'
                elif bet['type'] == 'CARDS': icon = '🟨'
                
                with cols[i]:
                    st.info(f"{bet['match']}\n\n**{icon} {bet['desc']}**")
                    st.metric("Quota", f"{bet['q']:.2f}", f"{bet['prob']*100:.0f}%")
                    st.write(f"💶 €{bet['stake']}")
        
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
                            
                            # 1X2 - LOGICA VERDE RISTRETTA
                            st.markdown("##### ⚖️ Esito Finale")
                            c1, c2, c3 = st.columns(3)
                            
                            def show_box(col, label, data):
                                is_green = (data['q'] <= 5.0 and data['roi'] >= 0.15)
                                style = "value-box" if is_green else "neutral-box"
                                val_txt = f"+{data['roi']*100:.0f}%" if data['roi'] > 0 else "-"
                                col.markdown(f"<div class='{style}'>{label}<br>{data['q']:.2f}<br><small>{val_txt}</small></div>", unsafe_allow_html=True)
                                
                            show_box(c1, "1", m['1x2']['1']); show_box(c2, "X", m['1x2']['X']); show_box(c3, "2", m['1x2']['2'])
                            
                            st.divider()
                            
                            with st.expander("⚔️ Testa a Testa (Stats)"):
                                h2h_cols = st.columns(4)
                                for idx, h in enumerate(m['h2h']):
                                    with h2h_cols[idx]:
                                        st.markdown(f"**{h['metric']}**")
                                        st.write(f"{h['fav']} {h['prob']*100:.0f}%")

                            def show_props(title, bets, exp):
                                if bets:
                                    st.markdown(f"#### {title} <span class='small-text'>(Att: {exp[0]:.1f} vs {exp[1]:.1f})</span>", unsafe_allow_html=True)
                                    cols = st.columns(len(bets))
                                    for idx, p in enumerate(bets):
                                        with cols[idx]:
                                            st.caption(f"{p['desc']} (@{p['q']:.2f})")
                                            st.markdown(custom_progress(p['prob']), unsafe_allow_html=True)
                                            st.markdown(f"<div class='prob-text'>Prob: {p['prob']*100:.0f}%</div>", unsafe_allow_html=True)

                            show_props("⚽ Gol", m['gol_bets'], m['exp_gol'])
                            show_props("🚩 Corner", m['corn_bets'], m['exp_corn'])
                            show_props("🛑 Falli", m['foul_bets'], m['exp_foul'])
                            show_props("🎯 Tiri Porta", m['shots_bets'], m['exp_shots'])
                            show_props("🟨 Cartellini", m['cards_bets'], m['exp_cards'])
                            
                            # --- SEZIONE TERMINALE (RAW) ---
                            st.markdown("---")
                            with st.expander("📟 Vista Terminale (Dati Grezzi)"):
                                st.code(m['raw_text'], language='text')
