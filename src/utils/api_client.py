import os
import openai
import google.generativeai as genai
from typing import Dict, List, Optional, Any
import json
import pandas as pd
import numpy as np

from .config import get_config

class APIClient:
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config()
        self.openai_client = None
        self.gemini_model = None
        self._setup_apis()
    
    def _setup_apis(self):
        """Setup API clients using configuration"""
        api_config = self.config_manager.get_api_config()
        
        # OpenAI setup - try config first, then environment
        openai_key = api_config.openai or os.getenv('OPENAI_API_KEY')
        if openai_key:
            openai.api_key = openai_key
            self.openai_client = openai
            print("OpenAI API configured successfully")
        else:
            print("OpenAI API key not found in config or environment variables")
        
        # Gemini setup - try config first, then environment
        gemini_key = api_config.gemini or os.getenv('GEMINI_API_KEY')
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
            print("Gemini API configured successfully")
        else:
            print("Gemini API key not found in config or environment variables")
    
    def analyze_market_sentiment(self, news_data: List[str], method: str = 'gemini') -> Dict[str, float]:
        """Analyze market sentiment from news data"""
        
        if method == 'gemini' and self.gemini_model:
            return self._analyze_sentiment_gemini(news_data)
        elif method == 'openai' and self.openai_client:
            return self._analyze_sentiment_openai(news_data)
        else:
            print(f"API method {method} not available, returning neutral sentiment")
            return {'sentiment_score': 0.0, 'confidence': 0.5}
    
    def _analyze_sentiment_gemini(self, news_data: List[str]) -> Dict[str, float]:
        """Analyze sentiment using Gemini API"""
        try:
            # Combine news articles
            combined_text = " ".join(news_data[:10])  # Limit to first 10 articles
            
            prompt = f"""
            Analyze the market sentiment of the following financial news articles.
            Return a JSON response with:
            - sentiment_score: float between -1 (very negative) and 1 (very positive)
            - confidence: float between 0 and 1 indicating confidence in the analysis
            - key_themes: list of main themes identified
            
            News articles:
            {combined_text}
            
            Respond only with valid JSON.
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {'sentiment_score': 0.0, 'confidence': 0.5, 'key_themes': []}
                
        except Exception as e:
            print(f"Error in Gemini sentiment analysis: {e}")
            return {'sentiment_score': 0.0, 'confidence': 0.5, 'key_themes': []}
    
    def _analyze_sentiment_openai(self, news_data: List[str]) -> Dict[str, float]:
        """Analyze sentiment using OpenAI API"""
        try:
            combined_text = " ".join(news_data[:10])
            
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyst. Analyze market sentiment and respond with JSON only."},
                    {"role": "user", "content": f"""
                    Analyze the market sentiment of these financial news articles.
                    Return JSON with:
                    - sentiment_score: float between -1 (very negative) and 1 (very positive)
                    - confidence: float between 0 and 1
                    - key_themes: list of main themes
                    
                    News: {combined_text}
                    """}
                ],
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error in OpenAI sentiment analysis: {e}")
            return {'sentiment_score': 0.0, 'confidence': 0.5, 'key_themes': []}
    
    def generate_trading_insights(self, market_data: Dict, predictions: Dict, 
                                method: str = 'gemini') -> Dict[str, Any]:
        """Generate trading insights using AI"""
        
        if method == 'gemini' and self.gemini_model:
            return self._generate_insights_gemini(market_data, predictions)
        elif method == 'openai' and self.openai_client:
            return self._generate_insights_openai(market_data, predictions)
        else:
            return {'insights': 'AI analysis not available', 'risk_assessment': 'medium'}
    
    def _generate_insights_gemini(self, market_data: Dict, predictions: Dict) -> Dict[str, Any]:
        """Generate insights using Gemini"""
        try:
            prompt = f"""
            As a quantitative trading analyst, provide insights based on this data:
            
            Market Data Summary:
            {json.dumps(market_data, indent=2)}
            
            Model Predictions:
            {json.dumps(predictions, indent=2)}
            
            Provide JSON response with:
            - insights: key trading insights (string)
            - risk_assessment: overall risk level (low/medium/high)
            - recommended_actions: list of recommended actions
            - market_outlook: short-term market outlook
            
            Focus on actionable insights for algorithmic trading.
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                return {
                    'insights': response.text,
                    'risk_assessment': 'medium',
                    'recommended_actions': [],
                    'market_outlook': 'neutral'
                }
                
        except Exception as e:
            print(f"Error generating Gemini insights: {e}")
            return {'insights': 'Analysis unavailable', 'risk_assessment': 'medium'}
    
    def _generate_insights_openai(self, market_data: Dict, predictions: Dict) -> Dict[str, Any]:
        """Generate insights using OpenAI"""
        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a quantitative trading analyst providing actionable insights."},
                    {"role": "user", "content": f"""
                    Analyze this trading data and provide insights:
                    
                    Market Data: {json.dumps(market_data)}
                    Predictions: {json.dumps(predictions)}
                    
                    Return JSON with insights, risk_assessment, recommended_actions, and market_outlook.
                    """}
                ],
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error generating OpenAI insights: {e}")
            return {'insights': 'Analysis unavailable', 'risk_assessment': 'medium'}
    
    def optimize_portfolio_allocation(self, portfolio_data: Dict, method: str = 'gemini') -> Dict[str, float]:
        """Get AI-assisted portfolio optimization suggestions"""
        
        if method == 'gemini' and self.gemini_model:
            return self._optimize_allocation_gemini(portfolio_data)
        elif method == 'openai' and self.openai_client:
            return self._optimize_allocation_openai(portfolio_data)
        else:
            # Return equal weight allocation as fallback
            symbols = list(portfolio_data.get('positions', {}).keys())
            if symbols:
                weight = 1.0 / len(symbols)
                return {symbol: weight for symbol in symbols}
            return {}
    
    def _optimize_allocation_gemini(self, portfolio_data: Dict) -> Dict[str, float]:
        """Portfolio optimization using Gemini"""
        try:
            prompt = f"""
            As a portfolio manager, optimize this portfolio allocation:
            
            Current Portfolio:
            {json.dumps(portfolio_data, indent=2)}
            
            Provide optimal allocation weights (sum to 1.0) in JSON format:
            {{"AAPL": 0.25, "MSFT": 0.30, ...}}
            
            Consider risk-return tradeoffs and diversification.
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            try:
                result = json.loads(response.text)
                # Normalize weights to sum to 1
                total_weight = sum(result.values())
                if total_weight > 0:
                    return {k: v/total_weight for k, v in result.items()}
                return result
            except:
                return {}
                
        except Exception as e:
            print(f"Error in Gemini portfolio optimization: {e}")
            return {}
    
    def _optimize_allocation_openai(self, portfolio_data: Dict) -> Dict[str, float]:
        """Portfolio optimization using OpenAI"""
        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a portfolio optimization expert."},
                    {"role": "user", "content": f"""
                    Optimize portfolio allocation for: {json.dumps(portfolio_data)}
                    
                    Return JSON with symbol weights that sum to 1.0.
                    """}
                ],
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error in OpenAI portfolio optimization: {e}")
            return {}