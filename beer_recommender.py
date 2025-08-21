import pandas as pd
import numpy as np
import os
from groq import Groq
import json
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
import streamlit as st

class BeerRecommender:
    def __init__(self):
        self.df = None
        self.gb_model = None
        self.global_scaler_recommend = None
        self.scalar = None
        self.encoder = None
        self.encoder2 = None
        self.X_reg_scaled = None
        self.mainstream_patterns = [
            'co.', 'inc', 'budweiser', 'bud', 'busch', 'michelob',
            'miller', 'coors', 'keystone', 'blue moon',
            'pabst', 'pbr', 'schlitz', 'old milwaukee',
            'rolling rock', 'yuengling', 'natural light', 'natty',
            'samuel adams', 'sam adams', 'boston lager',
            'corona', 'modelo', 'pacifico',
            'dos equis', 'tecate', 'sol', 'victoria',
            'heineken', 'amstel', 'stella artois',
            'becks', "beck's", 'st pauli', 'warsteiner',
            'guinness', 'harp', 'smithwick', 'kilkenny',
            'peroni', 'moretti', 'nastro azzurro',
            'carlsberg', 'tuborg', 'kronenbourg',
            'fosters', "foster's", 'grolsch', 'pilsner urquell',
            'molson', 'labatt', 'moosehead', 'sleeman',
            'sapporo', 'asahi', 'kirin', 'tsingtao', 'singha', 'tiger', 'leo',
            'shock top', 'goose island', 'elysian', 'lagunitas',
            'ballast point', '10 barrel', 'golden road',
            'blue point', 'devils backbone', 'karbach',
            'breckenridge', 'four peaks', 'wicked weed',
            'sierra nevada', 'new belgium', 'fat tire',
            'stone', 'brooklyn', 'dogfish head', 
            "bell's", 'bells brewery', 'founders',
            'deschutes', 'rogue', 'anchor steam',
            'red stripe', 'newcastle', 'bass', 'boddingtons',
            'murphy', 'beamish', 'tennents', 'carling',
            'leinenkugel', 'magic hat', 'pyramid',
            'widmer', 'redhook', 'kona', 'longboard',
            'landshark', 'presidente', 'medalla',
            'kingfisher', 'haywards', 'thunderbolt',
            'kalyani', 'knockout', 'royal challenge',
            'carlsberg elephant', 'bira 91', 'bira',
            'simba', 'godfather', 'hunter', 'zingaro',
            'london pilsner', 'kotsberg', 'bullet',
            'khajuraho', 'taj mahal', 'flying horse', 'dansberg',
            'golden eagle', 'guru', 'bad monkey', 'bee young',
            'white rhino', 'white owl', 'effingut'
        ]
        self.scaling_features = ['ABV', 'Astringency', 'Body', 'Alcohol', 'Bitter',
                                 'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices', 'Malty']
        
    def load_and_preprocess_data(self):
        self.df = pd.read_csv('./data/beer_profile_and_ratings.csv')
        
        self.df['mainstream'] = self.df.apply(
            lambda row: self.matches_mainstream_pattern(row['Beer Name (Full)']), axis=1
        )
        self.df['mainstream'] = self.df['mainstream'] | (self.df['number_of_reviews'] >= 300)
        
        self.df['strength'] = self.df['ABV'].apply(
            lambda x: 'Light' if x <= 5 else
                      'Medium' if x <= 7 else
                      'Strong' if x <= 10 else
                      'Extra Strong'
        )
        
        # Load country data
        beer_countries = pd.read_csv('./beer_names_country_latest.csv')
        beer_countries = beer_countries.drop(columns=['Name'])
        
        # Merge with country data
        self.df = pd.merge(
            self.df,
            beer_countries,
            how="inner",
            on="Beer Name (Full)"
        )
        
        self.df = self.df.drop(columns=['Min IBU', 'Max IBU', 'review_aroma', 
                                        'review_appearance', 'review_palate', 
                                        'review_taste', 'Beer Name (Full)', 'Brewery'])
        
        cols = self.df.columns.tolist()
        cols[1], cols[2] = cols[2], cols[1]
        self.df = self.df[cols]
        self.df['mainstream'] = self.df['mainstream'].astype(int)
        
    def matches_mainstream_pattern(self, beer_name_full):
        combined_name = beer_name_full.lower()
        for pattern in self.mainstream_patterns:
            if pattern in combined_name:
                return True
        return False
    
    def train_regression_model(self):
        reg_df = self.df.drop(columns=['number_of_reviews', 'strength', 'Name', 'Description', 'Country'])
        
        cols = reg_df.columns.tolist()
        cols[-2], cols[-1] = cols[-1], cols[-2]
        reg_df = reg_df[cols]
        
        y_reg = reg_df.iloc[:, -1]
        X_reg = reg_df.iloc[:, :-1]
        
        X = X_reg.copy()
        
        self.scalar = MinMaxScaler()
        X[self.scaling_features] = self.scalar.fit_transform(X[self.scaling_features])
        
        X['Style'] = X['Style'].str.split(' - ').str[0].str.split(' / ').str[0]
        
        self.encoder = OneHotEncoder(sparse_output=False)
        encoded_array = self.encoder.fit_transform(X[['Style']])
        feature_names = self.encoder.get_feature_names_out(['Style'])
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=X_reg.index)
        
        self.X_reg_scaled = pd.concat([X.drop('Style', axis=1), encoded_df], axis=1)
        
        X_train = self.X_reg_scaled.to_numpy()
        y_train = y_reg.to_numpy()
        
        self.gb_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4
        )
        self.gb_model.fit(X_train, y_train)
        
        self.global_scaler_recommend = MinMaxScaler()
        self.global_scaler_recommend.fit(self.df[self.scaling_features])
        
    def get_beer_features_from_text(self, user_input):
        try:
            # Try to get API key from Streamlit secrets first (for deployment)
            if 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets['GROQ_API_KEY']
            else:
                # Fall back to environment variable (for local development)
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("GROQ_API_KEY")
            
            if not api_key:
                raise ValueError("GROQ API key not found. Please set it in .env file or Streamlit secrets.")
            
            client = Groq(api_key=api_key)
            
            system_prompt = """
                            You are a beer flavor profile translator. Convert natural language beer preferences into numerical flavor profiles.

                            ## Output Format
                            Return a JSON with these exact fields:
                            - ABV: (float) 0.0-57.5
                            - Astringency: (int) 0-81
                            - Body: (int) 0-175
                            - Alcohol: (int) 0-139
                            - Bitter: (int) 0-150
                            - Sweet: (int) 0-263
                            - Sour: (int) 0-284
                            - Salty: (int) 0-48
                            - Fruits: (int) 0-175
                            - Hoppy: (int) 0-172
                            - Spices: (int) 0-184
                            - Malty: (int) 0-239
                            - mainstream: (int) 0 or 1 (DEFAULT = 1)
                            - style: (string) Beer style category
                            - region: (string or null) Country of origin - DEFAULT = null (only use: "United States", "Belgium", "Germany", "United Kingdom", "Canada", "Japan", or null)

                            ## IMPORTANT: Mainstream Flag Rules
                            DEFAULT mainstream = 1 (always start with 1)

                            Only set mainstream = 0 when:
                            - Belgian styles mentioned (Tripel, Dubbel, Quad)
                            - Sour/Wild/Lambic/Brett explicitly mentioned
                            - Imperial/Dessert beers with ABV > 9
                            - User explicitly says "craft", "artisanal", "specialty"
                            - Highly experimental flavor combinations

                            Keep mainstream = 1 for:
                            - All standard styles (IPA, Pilsner, Lager, Wheat, Stout, Amber)
                            - Any request without special keywords above
                            - "Sessionable", "refreshing", "light" beers
                            - When in doubt, use mainstream = 1

                            ## IMPORTANT: Region/Country Rules
                            DEFAULT region = null (always start with null)

                            Only set region to one of these 6 countries when EXPLICITLY mentioned:
                            - "United States" (also for: US, USA, America, American, from United States)
                            - "Belgium" (also for: Belgian)
                            - "Germany" (also for: German, Deutschland)
                            - "United Kingdom" (also for: UK, British, England, English, from the United Kingdom)
                            - "Canada" (also for: Canadian)
                            - "Japan" (also for: Japanese)

                            Keep region = null when:
                            - No country/origin is mentioned
                            - User mentions a country not in the allowed list
                            - User mentions general terms like "imported", "foreign", "domestic"
                            - When in doubt, use null

                            Region mapping examples:
                            - "American IPA" → region: "United States"
                            - "Belgian tripel" → region: "Belgium"  
                            - "German pilsner" → region: "Germany"
                            - "I want a beer from Japan" → region: "Japan"
                            - "British ale" → region: "United Kingdom"
                            - "Canadian lager" → region: "Canada"
                            - "from United States" → region: "United States"
                            - "from the United Kingdom" → region: "United Kingdom"
                            - "French beer" → region: null (not in allowed list)
                            - "hoppy IPA" → region: null (no country mentioned)

                            ## Scaling Guidelines
                            Use percentages of max range:
                            - "Very low/minimal": 3-10%
                            - "Low/light": 10-25%
                            - "Moderate/medium": 25-45%
                            - "High": 50-70%
                            - "Very high": 70-85%
                            - "Extremely/maximum": 85-100%

                            ## Core Translation Rules

                            ### Intensity Modifiers
                            - No modifier = use style default or 30-50% range
                            - "Slightly/hint of" = reduce by 50%
                            - "Very" = 70-85% of max
                            - "Extremely/super" = 85-100% of max
                            - "No/without" = 5-10% of max

                            ### Strength/Alcohol Keywords
                            - "light" → ABV: 3.2-4.5, Body: 25-35 (15-20%), Alcohol: 10-20 (7-14%)
                            - "sessionable" → ABV: 4-5, Body: 30-40 (17-23%), Alcohol: 15-25
                            - "medium/regular" → ABV: 5-6, Body: 60-80 (34-46%), Alcohol: 40-70
                            - "strong" → ABV: 7-9, Body: 70-90, Alcohol: 75-100 (54-72%)
                            - "very strong/imperial" → ABV: 9-12, Body: 120-160, Alcohol: 100-130

                            ### Flavor Keywords
                            - "citrusy" → Fruits: 140 (80%), Sour: 85 (30%)
                            - "tropical" → Fruits: 145 (83%), Sour: 15 (5%)
                            - "orangey" → Fruits: 155 (89%), add Sour: 240 if "tart"
                            - "fruity" → Fruits: 120 (69%)
                            - "chocolate" → Spices: 140 (76%), Malty: 210 (88%)
                            - "coffee" → Spices: 140 (76%), Astringency: 55 (68%)
                            - "spicy" → Spices: 155 (84%)
                            - "funky/brett" → Sour: 265 (93%), Astringency: 65 (80%)
                            - "tart" → Sour: 240+ (85%+), Astringency: 45+ (56%+)

                            ### Hop/Bitter Keywords
                            - "hoppy" → Hoppy: 150 (87%), Bitter: 110 (73%)
                            - "very hoppy" → Hoppy: 155-165 (90-96%), Bitter: 120-135
                            - "bitter" → Bitter: 110-135 (73-90%)
                            - "no hops" → Hoppy: 20 (12%), Bitter: 20 (13%)

                            ### Sweet/Malty Keywords
                            - "sweet" → Sweet: 145 (55%)
                            - "very sweet" → Sweet: 195-210 (74-80%)
                            - "dessert" → Sweet: 195 (74%), Body: 160 (91%)
                            - "no sweetness/dry" → Sweet: 15 (6%)
                            - "malty" → Malty: 185 (77%)
                            - "very malty" → Malty: 210 (88%)
                            - "not too malty" → Malty: 60 (25%)

                            ## Style Templates

                            ### IPA (mainstream = 1)
                            Base: Hoppy: 155, Bitter: 110, ABV: 6.8, Body: 75, Malty: 75

                            ### Pilsner (mainstream = 1)
                            Base: Hoppy: 65, Bitter: 45, ABV: 4.5, Body: 30, Malty: 80

                            ### Wheat Beer (mainstream = 1)
                            Base: Hoppy: 45, Body: 35, ABV: 4.2, Fruits: 85, Sour: 65

                            ### Lager (mainstream = 1)
                            Base: Hoppy: 60, Bitter: 55, ABV: 5.0, Body: 50, Malty: 60

                            ### Stout (mainstream = 1 unless imperial)
                            Base: Body: 140, Malty: 180, ABV: 6.5, Hoppy: 35
                            Imperial: ABV: 10.5, Body: 160, mainstream = 0

                            ### Belgian Tripel (mainstream = 0)
                            Base: ABV: 9.0, Spices: 155, Fruits: 95, Sweet: 115

                            ### Sour/Wild Ale (mainstream = 0)
                            Base: Sour: 265, Astringency: 65, Hoppy: 20, mainstream = 0

                            ### Amber/Red Ale (mainstream = 1)
                            Base: Malty: 185, Sweet: 145, Body: 95, Bitter: 35

                            ### Light Beer (mainstream = 1)
                            Base: ABV: 3.2, Body: 25, all others low (10-30% range)

                            ## Processing Order
                            1. Identify style first (sets base template)
                            2. Apply strength modifiers (light/strong/sessionable)
                            3. Apply flavor descriptors (additive)
                            4. Apply negations last (no sweetness, etc.)
                            5. Check mainstream flag (default = 1 unless special style)
                            6. Check region flag (default = null unless explicitly mentioned)

                            ## Examples
                            "hoppy IPA" → Start with IPA template, already has high hoppy, region: null
                            "light beer" → Use Light Beer template, region: null
                            "Belgian tripel" → Use Tripel template, set mainstream = 0, region: "Belgium"
                            "dessert stout" → Stout template + high sweet/body, mainstream = 0, region: null
                            "German pilsner" → Pilsner template, region: "Germany"
                            "I want a strong lager from United States" → Lager template with strong modifiers, region: "United States"

                            ## Special Edge Cases

                            ### Explicitly Bad/Poor Quality Requests
                            When user explicitly asks for "bad", "terrible", "awful", "worst", "horrible", "disgusting", "undrinkable" beer:
                            - Set ALL features to minimum values (3-10% of max)
                            - ABV: 0.05-1.0
                            - All flavor features: 1-10% of their max values
                            - Astringency: 2-5
                            - Body: 10-15
                            - Alcohol: 10-15
                            - Bitter: 3-10
                            - Sweet: 10-20
                            - Sour: 3-10
                            - Salty: 0-2
                            - Fruits: 1-10
                            - Hoppy: 3-10
                            - Spices: 3-10
                            - Malty: 15-25
                            - mainstream: 1
                            - style: "Low Alcohol Beer" or "Light Beer"
                            - region: null

                            Examples:
                            - "Just a bad beer" → Minimal everything, region: null
                            - "Give me your worst beer" → Lowest possible values, region: null
                            - "I want a terrible beer" → Near-zero features, region: null
                            - "Something awful" → Minimum profile, region: null

                            ### Testing/Experimental Requests
                            If user mentions "test", "experiment", or asks for unusual combinations that would clearly conflict (e.g., "extremely sweet AND extremely bitter AND light body"), recognize this as potentially problematic and generate values that reflect the conflict.
                        """
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            raise Exception(f"Error calling GROQ API: {e}")
    
    def predict_rating(self, llm_output):
        test_point = {col: 0 for col in self.X_reg_scaled.columns}
        
        for feat in self.scaling_features:
            test_point[feat] = [llm_output[feat]]
        
        test_point['mainstream'] = llm_output['mainstream']
        
        style_column = f"Style_{llm_output['style']}"
        if style_column in self.X_reg_scaled.columns:
            test_point[style_column] = 1
        
        test_point = pd.DataFrame(test_point)
        test_point[self.scaling_features] = self.scalar.transform(test_point[self.scaling_features])
        test_point = test_point[self.X_reg_scaled.columns]
        
        test_point = test_point.to_numpy()
        predicted_rating = self.gb_model.predict(test_point.reshape(1, -1))[0]
        
        return predicted_rating
    
    def get_strength(self, ABV):
        if ABV <= 5:
            return 'Light'
        elif ABV <= 7:
            return 'Medium'
        elif ABV <= 10:
            return 'Strong'
        else:
            return 'Extra Strong'
    
    def get_quality_score(self, rating, num_reviews):
        return rating * (0.6 + 0.4 * np.log1p(num_reviews) / 10)
    
    def generate_test_point(self, llm_output, X_train, scalar, type):
        test_point = {col: 0 for col in X_train.columns}
        
        for feat in self.scaling_features:
            test_point[feat] = [llm_output[feat]]
        
        style_column = f"Style_{llm_output['style']}"
        if style_column in X_train.columns:
            test_point[style_column] = 1
        
        if type == 'Regressor':
            test_point['mainstream'] = llm_output['mainstream']
        
        test_point = pd.DataFrame(test_point)
        test_point[self.scaling_features] = scalar.transform(test_point[self.scaling_features])
        test_point = test_point[X_train.columns]
        
        return test_point
    
    def get_beer_recommendations(self, llm_output, alt=False, alt_rating_threshold=3.0):
        X_recommend = self.df[['Style'] + self.scaling_features + ['mainstream', 'strength', 'Country']].copy()
        y_recommend = self.df[['Name', 'Description', 'review_overall', 'number_of_reviews']].copy()
        
        X_recommend['Style'] = X_recommend['Style'].str.split(' - ').str[0].str.split(' / ').str[0]
        
        self.encoder2 = OneHotEncoder(sparse_output=False)
        encoded_array = self.encoder2.fit_transform(X_recommend[['Style']])
        feature_names = self.encoder2.get_feature_names_out(['Style'])
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=X_recommend.index)
        X_recommend = pd.concat([X_recommend.drop('Style', axis=1), encoded_df], axis=1)
        
        # Filter by rating if alt recommendations
        if alt:
            rating_mask = y_recommend['review_overall'] >= alt_rating_threshold
            X_recommend = X_recommend[rating_mask]
            y_recommend = y_recommend[rating_mask]
        
        # Filter by region if specified
        if llm_output.get('region') is not None:
            country_mask = X_recommend['Country'].str.lower() == llm_output['region'].lower()
            X_recommend = X_recommend[country_mask]
            y_recommend = y_recommend[country_mask]
        
        # Filter by mainstream
        if llm_output['mainstream'] == 1:
            mainstream_mask = X_recommend['mainstream'] == 1
            X_recommend = X_recommend[mainstream_mask]
            y_recommend = y_recommend[mainstream_mask]
        
        # Filter by strength
        strength = self.get_strength(llm_output['ABV'])
        strength_mask = X_recommend['strength'] == strength
        X_recommend_sub = X_recommend[strength_mask]
        y_recommend_sub = y_recommend[strength_mask]
        
        # Drop columns used for filtering
        X_recommend_sub = X_recommend_sub.drop(columns=['strength', 'mainstream', 'Country'])
        
        # Scale features
        X_recommend_sub[self.scaling_features] = self.global_scaler_recommend.transform(
            X_recommend_sub[self.scaling_features]
        )
        
        X_recommend_scaled = X_recommend_sub
        X_recommend_scaled_np = X_recommend_scaled.to_numpy()
        y_recommend_np = y_recommend_sub.to_numpy()
        
        # If no beers match the criteria, return empty list
        if len(X_recommend_scaled_np) == 0:
            return []
        
        test_point_recommendation = self.generate_test_point(
            llm_output, X_recommend_scaled, self.global_scaler_recommend, type="Recommend"
        )
        test_point_recommendation_np = test_point_recommendation.values[0]
        
        # Use min of 10 or number of available beers
        n_neighbors = min(5, len(X_recommend_scaled_np))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(X_recommend_scaled_np)
        
        distances, indices = knn.kneighbors([test_point_recommendation_np])
        
        top_beers = []
        for i, idx in enumerate(indices[0]):
            beer_info = {
                'name': y_recommend_np[idx][0],
                'description': y_recommend_np[idx][1],
                'rating': y_recommend_np[idx][2],
                'num_reviews': y_recommend_np[idx][3],
                'distance': distances[0][i],
                'index': idx
            }
            top_beers.append(beer_info)
        
        for beer in top_beers:
            beer['quality_score'] = self.get_quality_score(beer['rating'], beer['num_reviews'])
        
        top_beers.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return top_beers[:3]
    
    def get_recommendations(self, user_input, region=None):
        # Get LLM features
        llm_output = self.get_beer_features_from_text(user_input)
        
        # Override region if specified by user in UI
        if region is not None:
            llm_output['region'] = region
        elif 'region' not in llm_output:
            llm_output['region'] = None
        
        predicted_rating = self.predict_rating(llm_output)
        
        # Get regular recommendations
        recommendations = self.get_beer_recommendations(llm_output, alt=False)
        
        # Get alternative recommendations if rating is low
        alt_recommendations = None
        if predicted_rating < 3.0:
            alt_recommendations = self.get_beer_recommendations(llm_output, alt=True, alt_rating_threshold=3.0)
        
        return {
            'predicted_rating': predicted_rating,
            'recommendations': recommendations,
            'alt_recommendations': alt_recommendations,
            'user_features': llm_output
        }
