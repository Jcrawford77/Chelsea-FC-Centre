import streamlit as st
import pandas as pd
import soccerdata as sd
import pulp
import plotly.graph_objects as go
from soccerplots.radar_chart import Radar
import matplotlib.pyplot as plt

st.set_page_config(page_title="Chelsea Player Stats", layout="wide")

# --- Create columns for title and badge ---
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Chelsea FC Centre")

with col2:
    st.image(
        "https://static.vecteezy.com/system/resources/previews/026/135/387/original/chelsea-club-logo-black-and-white-symbol-premier-league-football-abstract-design-illustration-free-vector.jpg",
        width=150
    )

# --- Tabs ---
tab_home, tab_radars, tab_shots, tab_match, tab_fpl, tab_optimal = st.tabs(
    ["Home", "Player Radars", "Shot Maps", "Match Analysis", "FPL", "FPL Optimal Squad"]
)

# --- Define URL mapping for data loading ---
url_mapping = {
    ("2024-2025", "FA Cup"): "https://fbref.com/en/squads/cff3d9bb/2024-2025/c514/Chelsea-Stats-FA-Cup",
    ("2024-2025", "League Cup"): "https://fbref.com/en/squads/cff3d9bb/2024-2025/c690/Chelsea-Stats-EFL-Cup",
    ("2024-2025", "UEFA Conference League"): "https://fbref.com/en/squads/cff3d9bb/2024-2025/c882/Chelsea-Stats-Conference-League",
    ("2024-2025", "Premier League"): "https://fbref.com/en/squads/cff3d9bb/2024-2025/Chelsea-Stats",
    ("2025-2026", "FA Cup"): "https://fbref.com/en/squads/cff3d9bb/2025-2026/514/Chelsea-Stats-FA-Cup",
    ("2025-2026", "League Cup"): "https://fbref.com/en/squads/cff3d9bb/2025-2026/10/Chelsea-Stats-EFL-Cup",
    ("2025-2026", "UEFA Champions League"): "https://fbref.com/en/squads/cff3d9bb/2025-2026/19/Chelsea-Stats-Champions-League",
    ("2025-2026", "FIFA Club World Cup"): "https://fbref.com/en/squads/cff3d9bb/2025/Chelsea-Stats",
    ("2025-2026", "Premier League"): "https://fbref.com/en/squads/cff3d9bb/2025-2026/Chelsea-Stats",
}

# --- Helper function to load a single competition's data (for Chelsea)
@st.cache_data
def load_single_competition_data(season, competition):
    try:
        df = pd.DataFrame()
        fbref_url = url_mapping.get((season, competition))
        if fbref_url:
            # Use headers to mimic a web browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            tables = pd.read_html(fbref_url, storage_options={'headers': headers})
            if not tables:
                return pd.DataFrame()
            df = tables[0]

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns]
            
            df.columns = [c.replace('Unnamed: ', '').replace(' ', '_').replace('%', 'pct').lower() for c in df.columns]
            df.dropna(how='all', inplace=True)
            
            rename_map = {}
            for col in df.columns:
                if 'player' in col and col != 'player': rename_map[col] = 'player'
                elif 'gls' in col: rename_map[col] = 'goals'
                elif 'ast' in col: rename_map[col] = 'assists'
                elif 'mp' in col: rename_map[col] = 'apps'
                elif col == 'min': rename_map[col] = 'playing_time_min'
                elif 'xg' in col and '90' in col: rename_map[col] = 'per_90_minutes_xg'
                elif 'xag' in col and '90' in col: rename_map[col] = 'per_90_minutes_xag'
                elif 'pos' in col: rename_map[col] = 'pos'
                elif 'nation' in col: rename_map[col] = 'nation'
                elif 'age' in col: rename_map[col] = 'age'
                
            df.rename(columns=rename_map, inplace=True)
            df = df.loc[:, ~df.columns.duplicated()]

            numeric_cols_to_clean = ['apps', 'playing_time_min', 'goals', 'assists', 'per_90_minutes_xg', 'per_90_minutes_xag', 'age']
            for col in numeric_cols_to_clean:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['season'] = season
            df['competition'] = competition
            
            position_order = ['GK', 'DF', 'MF', 'FW']
            if 'pos' in df.columns:
                df['pos'] = df['pos'].fillna('Other')
                df['pos'] = df['pos'].apply(lambda x: x.split(',')[0].strip() if x != 'Other' else 'Other')
                df['pos'] = pd.Categorical(df['pos'], categories=position_order + ['Other'], ordered=True)
                df.sort_values(by=['pos', 'player'], inplace=True)
                
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data for {competition}: {e}")
        return pd.DataFrame()

# --- Function to load all Chelsea data for consistent range calculation
@st.cache_data
def load_all_chelsea_data(season):
    all_competitions_urls = {
        "2024-2025": ["Premier League", "FA Cup", "League Cup", "UEFA Conference League"],
        "2025-2026": ["Premier League", "FA Cup", "League Cup", "UEFA Champions League", "FIFA Club World Cup"]
    }
    df_list = []
    for comp in all_competitions_urls[season]:
        df_comp = load_single_competition_data(season, comp)
        if not df_comp.empty:
            df_list.append(df_comp)
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    else:
        st.warning(f"Could not load any data for the {season} season.")
        return pd.DataFrame()

# --- Main `load_data` function now handles 'All Competitions' by aggregating
@st.cache_data
def load_data(season, competition):
    if competition == "All Competitions":
        df_all = load_all_chelsea_data(season)
        if df_all.empty:
            return pd.DataFrame()
        agg_funcs = {
            'apps': 'sum',
            'playing_time_min': 'sum',
            'goals': 'sum',
            'assists': 'sum',
            'per_90_minutes_xg': 'mean', 
            'per_90_minutes_xag': 'mean',
            'pos': lambda x: '/'.join(x.dropna().unique()),
            'nation': lambda x: '/'.join(x.dropna().unique()),
            'age': 'first',
            'team': 'first'
        }
        df_agg = df_all.groupby('player').agg(agg_funcs).reset_index()
        df_agg['season'] = season
        df_agg['competition'] = "All Competitions"
        return df_agg
    else:
        return load_single_competition_data(season, competition)

# --- NEW FUNCTION TO LOAD ALL LEAGUE DATA (for radars) ---
@st.cache_data
def load_league_data(league_id, season):
    try:
        fbref = sd.FBref(leagues=[league_id], seasons=[season])
        
        tables = {
            'standard': None,
            'misc': None
        }
        
        for stat_type in tables.keys():
            try:
                tables[stat_type] = fbref.read_player_season_stats(stat_type=stat_type).reset_index()
            except Exception as e:
                tables[stat_type] = pd.DataFrame() 

        df = tables['standard']
        if df.empty:
            return pd.DataFrame()

        if not tables['misc'].empty:
            df = pd.merge(df, tables['misc'], on=['player', 'team', 'season'], how='outer', suffixes=('', '_y'))
            df.drop(columns=[c for c in df.columns if '_y' in c], inplace=True)
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns]
        
        df.columns = [c.replace('Unnamed: ', '').replace(' ', '_').replace('%', 'pct').lower() for c in df.columns]
        
        rename_map = {}
        for col in df.columns:
            if 'player' in col and col != 'player': rename_map[col] = 'player'
            elif 'gls' in col: rename_map[col] = 'goals'
            elif 'ast' in col: rename_map[col] = 'assists'
            elif 'mp' in col: rename_map[col] = 'apps'
            elif col == 'min': rename_map[col] = 'playing_time_min'
            elif 'xg' in col and '90' in col: rename_map[col] = 'per_90_minutes_xg'
            elif 'xag' in col and '90' in col: rename_map[col] = 'per_90_minutes_xag'
            elif 'pos' in col: rename_map[col] = 'pos'
            elif 'nation' in col: rename_map[col] = 'nation'
            elif 'age' in col: rename_map[col] = 'age'
            elif 'team' in col and col != 'team': rename_map[col] = 'team'

        new_rename_map = {
            'playing_time_starts': 'starts',
            'playing_time_min': 'minutes_played',
            'playing_time_mp': 'appearances',
            'performance_gls': 'goals',
            'performance_ast': 'assists',
            'performance_pk': 'penalty_kicks',
            'performance_pkatt': 'penalty_kicks_attempted',
            'performance_crdy': 'yellow_cards',
            'performance_crdr': 'red_cards',
            'expected_xg': 'xg',
            'expected_xag': 'xag',
            'expected_npxg': 'npxg',
            'progression_prgc': 'progressive_carries',
            'progression_prgp': 'progressive_passes',
            'g-pk_per_90': 'goals_without_penalties_per_90'
        }
        rename_map.update(new_rename_map)
        
        df.rename(columns=rename_map, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        
        numeric_cols_to_clean = list(rename_map.values())
        for col in numeric_cols_to_clean:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['league'] = league_id
        
        return df
    except Exception as e:
        st.error(f"Error loading data for {league_id} league: {e}")
        return pd.DataFrame()


# --- Global selection for Chelsea data across all tabs ---
with st.sidebar:
    st.header("Global Data Selection")
    season_choice = st.selectbox("Select Season:", ["2024-2025", "2025-2026"], index=0, key="global_season")
    competition_options = {
        "2024-2025": ["Premier League", "FA Cup", "League Cup", "UEFA Conference League", "All Competitions"],
        "2025-2026": ["Premier League", "FA Cup", "League Cup", "UEFA Champions League", "FIFA Club World Cup", "All Competitions"]
    }
    
    competition_choice = st.selectbox("Select Competition:", competition_options[season_choice], key="global_competition")

df = load_data(season_choice, competition_choice)
all_season_chelsea_data = load_all_chelsea_data(season_choice)

# --- Home Tab ---
with tab_home:
    st.header("Chelsea Player Stats")
    if df.empty:
        st.warning("No data available.")
    else:
        st.subheader(f"Chelsea Player Stats - {competition_choice} ({season_choice})")
        st.dataframe(df)

# --- Player Radars Tab ---
with tab_radars:
    st.header("Player Radars")
    st.write("Compare player stats across different seasons and clubs.")
    
    params = ['minutes_played', 'appearances', 'starts', 'goals', 'assists', 'penalty_kicks', 'penalty_kicks_attempted', 'yellow_cards', 'red_cards', 'xg', 'xag', 'npxg', 'progressive_carries', 'progressive_passes']
    
    display_names_map = {
        'minutes_played': 'Minutes Played',
        'appearances': 'Appearances',
        'starts': 'Starts',
        'goals': 'Goals',
        'assists': 'Assists',
        'penalty_kicks': 'Penalty Kicks Scored',
        'penalty_kicks_attempted': 'Penalty Kicks Attempted',
        'yellow_cards': 'Yellow Cards',
        'red_cards': 'Red Cards',
        'xg': 'xG',
        'xag': 'xA',
        'npxg': 'Non-Penalty xG',
        'progressive_carries': 'Progressive Carries',
        'progressive_passes': 'Progressive Passes'
    }

    col1, col2 = st.columns(2)
    
    # --- Player 1 Selection ---
    with col1:
        st.subheader("Player 1")
        league1_choice = st.selectbox("Select League:", ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"], key="radar_league1")
        season1_choice = st.selectbox("Select Season:", ["2024-2025", "2025-2026"], key="radar_season1")
        
        df1 = load_league_data(league1_choice, season1_choice)
        
        if not df1.empty:
            clubs1_list = sorted(df1['team'].dropna().unique().tolist())
            if clubs1_list:
                club1 = st.selectbox("Select Club:", clubs1_list, key="radar_club1")
            
                df1_filtered = df1[df1['team'] == club1].copy()
            
                if 'pos' in df1_filtered.columns:
                    position_options = ["All", "GK", "DF", "MF", "FW"]
                    selected_pos1 = st.selectbox("Filter by Position:", position_options, key="pos_p1")
                    
                    if selected_pos1 != "All":
                        df1_filtered = df1_filtered[df1_filtered['pos'].astype(str).str.contains(selected_pos1, na=False)]
                
                players1_list = sorted(df1_filtered['player'].dropna().unique().tolist())
                if not players1_list:
                    st.warning("No players found for this filter.")
                    player1 = None
                else:
                    player1 = st.selectbox("Select Player 1:", players1_list, key="player1")
            else:
                st.warning("No clubs found for this league/season.")
                player1 = None
        else:
            st.warning("No data available for these selections.")
            player1 = None

    # --- Player 2 Selection ---
    with col2:
        st.subheader("Player 2")
        league2_choice = st.selectbox("Select League:", ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"], key="radar_league2")
        season2_choice = st.selectbox("Select Season:", ["2024-2025", "2025-2026"], key="radar_season2")
        
        df2 = load_league_data(league2_choice, season2_choice)
        
        if not df2.empty:
            clubs2_list = sorted(df2['team'].dropna().unique().tolist())
            if clubs2_list:
                club2 = st.selectbox("Select Club:", clubs2_list, key="radar_club2")
            
                df2_filtered = df2[df2['team'] == club2].copy()
            
                if 'pos' in df2_filtered.columns:
                    position_options = ["All", "GK", "DF", "MF", "FW"]
                    selected_pos2 = st.selectbox("Filter by Position:", position_options, key="pos_p2")
                    
                    if selected_pos2 != "All":
                        df2_filtered = df2_filtered[df2_filtered['pos'].astype(str).str.contains(selected_pos2, na=False)]
                
                players2_list = sorted(df2_filtered['player'].dropna().unique().tolist())
                if not players2_list:
                    st.warning("No players found for this filter.")
                    player2 = None
                else:
                    player2 = st.selectbox("Select Player 2:", players2_list, key="player2")
            else:
                st.warning("No clubs found for this league/season.")
                player2 = None
        else:
            st.warning("No data available for these selections.")
            player2 = None

    if player1 and player2:
        p1_data = df1[(df1['player'] == player1) & (df1['team'] == club1)]
        p2_data = df2[(df2['player'] == player2) & (df2['team'] == club2)]
        
        if not p1_data.empty and not p2_data.empty:
            
            all_leagues = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
            all_seasons = ["2024-2025", "2025-2026"]
            all_data_for_ranges = pd.concat([load_league_data(league, season) for league in all_leagues for season in all_seasons], ignore_index=True)
            
            ranges = []
            for col in params:
                if col in all_data_for_ranges.columns and not all_data_for_ranges[col].isnull().all():
                    min_val = all_data_for_ranges[col].min()
                    max_val = all_data_for_ranges[col].max()
                    ranges.append((min_val, max_val))
                else:
                    ranges.append((0, 1))
            
            p1_values = []
            p2_values = []
            for col in params:
                if col in p1_data.columns:
                    p1_values.append(p1_data[col].values[0])
                else:
                    p1_values.append(0)
                
                if col in p2_data.columns:
                    p2_values.append(p2_data[col].values[0])
                else:
                    p2_values.append(0)
            
            values = [p1_values, p2_values]
            
            radar_params = [display_names_map.get(p, p) for p in params]

            title = dict(
                title_name='', title_colour='red', subtitle_name='', subtitle_colour='red',
                title_name_2='', title_colour_2='blue', subtitle_name_2='', subtitle_colour_2='blue',
                title_fontsize=18,
            )
            
            # --- START OF CODE MODIFICATION ---
            # Set the path to your image file
            logo_path = 'Bogey Data logo.png' # <--- Change this to your image file name

            # Adjust these coordinates to place the logo where you want it
            # The format is [left, bottom, width, height] as a percentage of the plot area
            image_coords = [0.475, 0.435, 0.05, 0.05]
            
            radar = Radar()
            fig, ax = radar.plot_radar(
                ranges=ranges,
                params=radar_params,
                values=values,
                radar_color=['red', 'blue'],
                alphas=[.75, .6],
                title=title,
                endnote=f'Made by Bogey Data',
                compare=True,
                image=logo_path,
                image_coord=image_coords
            )
            # --- END OF CODE MODIFICATION ---
            
            ax.text(0.01, 1.05, player1, transform=ax.transAxes, fontsize=18, color='red', ha='left', va='top', fontweight='bold')
            ax.text(0.01, 1.02, club1, transform=ax.transAxes, fontsize=14, color='red', ha='left', va='top')
            
            ax.text(0.99, 1.05, player2, transform=ax.transAxes, fontsize=18, color='blue', ha='right', va='top', fontweight='bold')
            ax.text(0.99, 1.02, club2, transform=ax.transAxes, fontsize=14, color='blue', ha='right', va='top')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.error("Could not generate radar plot. Check if both players have data for their selected league/club.")


                










# --- Shot Maps Tab ---
with tab_shots:
    st.header("Shot Maps")
    st.write("Add shot map visualizations here.")

# --- Match Analysis Tab ---
with tab_match:
    st.header("Match Analysis")
    st.write("Add match performance analytics here.")

# --- FPL Tab ---
with tab_fpl:
    st.header("Fantasy Premier League")

    # --- Load CSV ---
    fpl_data = pd.read_csv("fpl.csv")

    # --- Clean and format columns ---
    fpl_data["sel%"] = fpl_data["sel%"].str.rstrip('%').astype(float)
    fpl_data["pts_per_million"] = fpl_data["pts"] / fpl_data["cost"]

    # --- Position order for sorting ---
    position_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    fpl_data["PositionOrder"] = fpl_data["pos"].map(position_order)

    # --- Dropdown to select team ---
    teams = sorted(fpl_data["club"].unique())
    selected_team = st.selectbox("Select team", ["All"] + teams)

    if selected_team != "All":
        team_data = fpl_data[fpl_data["club"] == selected_team].copy()
        team_data.sort_values(["PositionOrder", "pts"], ascending=[True, False], inplace=True)
    else:
        team_data = fpl_data.copy()
        team_data.sort_values("pts_per_million", ascending=False, inplace=True)
        team_data = team_data.head(30)

    team_data.reset_index(drop=True, inplace=True)

    # --- Highlight positions ---
    def color_positions(row):
        color_map = {"GKP": "lightblue", "DEF": "lightgreen", "MID": "lightyellow", "FWD": "lightpink"}
        return [f'background-color: {color_map[row.pos]}' for _ in row]

    top3_index = team_data["pts_per_million"].nlargest(3).index

    def highlight_top3_rows(row):
        if row.name in top3_index:
            return ['background-color: gold'] * len(row)
        return [''] * len(row)

    styled_df = (
        team_data.style
        .format({"sel%": "{:.1f}%", "cost": "{:.1f}", "pts_per_million": "{:.1f}"})
        .apply(color_positions, axis=1)
        .apply(highlight_top3_rows, axis=1)
    )

    st.dataframe(styled_df)

    # --- Top 3 pts per million ---
    top3_players = team_data.loc[top3_index, ["player", "pts_per_million"]]
    st.write("🏆 Top 3 Points per Million Players:")
    for idx, row in top3_players.iterrows():
        st.write(f"{row['player']}: {row['pts_per_million']:.1f}")

# --- FPL Optimal Squad Tab ---
with tab_optimal:
    # --- Prepare FPL data ---
    fpl_data["pts"] = pd.to_numeric(fpl_data["pts"], errors="coerce")
    fpl_data["cost"] = pd.to_numeric(fpl_data["cost"], errors="coerce")
    fpl_data.dropna(subset=["pts", "cost"], inplace=True)

    # --- Define LP problem ---
    prob = pulp.LpProblem("FPL_MaxPoints", pulp.LpMaximize)

    # --- Decision variables ---
    player_vars = pulp.LpVariable.dicts("Player", fpl_data.index, cat="Binary")

    # --- Objective ---
    prob += pulp.lpSum([player_vars[i] * fpl_data.loc[i, "pts"] for i in fpl_data.index])

    # --- Constraints ---
    prob += pulp.lpSum([player_vars[i] for i in fpl_data.index]) == 15
    position_limits = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for pos, limit in position_limits.items():
        prob += pulp.lpSum([player_vars[i] for i in fpl_data.index if fpl_data.loc[i, "pos"] == pos]) == limit

    prob += pulp.lpSum([player_vars[i] * fpl_data.loc[i, "cost"] for i in fpl_data.index]) <= 100

    # --- Solve LP ---
    prob.solve()

    # --- Extract squad ---
    squad_indices = [i for i in fpl_data.index if player_vars[i].value() == 1]
    squad = fpl_data.loc[squad_indices].copy()

    # --- Pick starters and bench ---
    def pick_starters(df):
        starters = []
        bench = []

        gk = df[df.pos == "GKP"].nlargest(1, "pts")
        starters.append(gk)

        def_ = df[df.pos == "DEF"].nlargest(4, "pts")
        starters.append(def_)

        mid = df[df.pos == "MID"].nlargest(4, "pts")
        starters.append(mid)

        fwd = df[df.pos == "FWD"].nlargest(2, "pts")
        starters.append(fwd)

        starters_df = pd.concat(starters)

        remaining = df.drop(starters_df.index)
        bench_gk = remaining[remaining.pos == "GKP"].nsmallest(1, "cost")
        bench_def = remaining[remaining.pos == "DEF"].nsmallest(1, "cost")
        bench_mid = remaining[remaining.pos == "MID"].nsmallest(1, "cost")
        bench_fwd = remaining[remaining.pos == "FWD"].nsmallest(1, "cost")
        bench_df = pd.concat([bench_gk, bench_def, bench_mid, bench_fwd])

        return starters_df, bench_df

    starters_df, bench_df = pick_starters(squad)

    final_squad = pd.concat([starters_df, bench_df])
    final_squad["PositionOrder"] = final_squad["pos"].map(position_order)
    final_squad.sort_values(["PositionOrder", "pts"], ascending=[True, False], inplace=True)

    st.subheader("Optimal Raw Points Squad")
    st.dataframe(final_squad[["player", "pos", "cost", "pts", "pts_per_million"]])

    total_points = final_squad["pts"].sum()
    total_cost = final_squad["cost"].sum()
    st.write(f"💰 Total Squad Cost: £{total_cost:.1f}m")
    st.write(f"🏆 Total Squad Points: {total_points:.0f}")

