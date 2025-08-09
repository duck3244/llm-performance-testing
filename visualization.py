"""
오픈소스 LLM 최적화 결과 시각화 및 분석 시스템
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ResultVisualizer:
    """최적화 결과 시각화 클래스"""

    def __init__(self, results_dir: str = "optimization_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # 시각화 저장 디렉토리
        self.viz_dir = Path("visualizations")
        self.viz_dir.mkdir(exist_ok=True)

    def load_optimization_results(self, pattern: str = "opt_*.json") -> List[Dict[str, Any]]:
        """최적화 결과 파일들 로드"""
        results = []

        for filepath in self.results_dir.glob(pattern):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"파일 {filepath} 로드 실패: {e}")

        return results

    def load_benchmark_results(self, pattern: str = "bench_*.json") -> List[Dict[str, Any]]:
        """벤치마크 결과 파일들 로드"""
        results = []

        for filepath in self.results_dir.glob(pattern):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"파일 {filepath} 로드 실패: {e}")

        return results

    def load_all_results(self) -> List[Dict[str, Any]]:
        """모든 결과 파일 로드"""
        return self.load_optimization_results("*.json")

    def create_optimization_analysis(self, results: List[Dict[str, Any]],
                                   save_path: str = None) -> go.Figure:
        """최적화 결과 분석 차트 생성"""
        if not results:
            return None

        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '모델별 최적화 성능',
                '파라미터 최적화 트렌드',
                '하드웨어 사용량 분석',
                '파라미터 상관관계'
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )

        # 1. 모델별 성능 막대 차트
        model_scores = {}
        for result in results:
            model = result.get('model_name', 'Unknown')
            score = result.get('best_score', 0)
            if model not in model_scores or score > model_scores[model]:
                model_scores[model] = score

        fig.add_trace(
            go.Bar(
                x=list(model_scores.keys()),
                y=list(model_scores.values()),
                name='최고 점수',
                marker_color='lightblue',
                showlegend=False
            ),
            row=1, col=1
        )

        # 2. 최적화 트렌드 (시간별)
        if len(results) > 1:
            timestamps = []
            scores = []
            models = []

            for result in sorted(results, key=lambda x: x.get('timestamp', '')):
                timestamps.append(result.get('timestamp', ''))
                scores.append(result.get('best_score', 0))
                models.append(result.get('model_name', 'Unknown'))

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(timestamps))),
                    y=scores,
                    mode='lines+markers',
                    name='성능 트렌드',
                    text=models,
                    hovertemplate='<b>%{text}</b><br>점수: %{y:.3f}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )

        # 3. 하드웨어 사용량 히트맵
        hw_data = []
        hw_models = []
        hw_metrics = ['cpu_avg', 'memory_peak', 'gpu_memory_peak']

        for result in results:
            if 'hardware_usage' in result:
                hw = result['hardware_usage']
                model = result.get('model_name', 'Unknown')
                hw_models.append(model)
                hw_data.append([
                    hw.get('cpu_avg', 0),
                    hw.get('memory_peak', 0),
                    hw.get('gpu_memory_peak', 0)
                ])

        if hw_data:
            hw_array = np.array(hw_data).T
            fig.add_trace(
                go.Heatmap(
                    z=hw_array,
                    x=hw_models,
                    y=['CPU (%)', '메모리 (%)', 'GPU 메모리 (GB)'],
                    colorscale='Viridis',
                    showscale=False
                ),
                row=2, col=1
            )

        # 4. 파라미터 상관관계 (Temperature vs Top-p)
        temperatures = []
        top_ps = []
        scores = []
        model_names = []

        for result in results:
            best_params = result.get('best_params', {})
            temperatures.append(best_params.get('temperature', 0))
            top_ps.append(best_params.get('top_p', 0))
            scores.append(result.get('best_score', 0))
            model_names.append(result.get('model_name', 'Unknown'))

        fig.add_trace(
            go.Scatter(
                x=temperatures,
                y=top_ps,
                mode='markers',
                marker=dict(
                    size=10,
                    color=scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="점수")
                ),
                text=model_names,
                hovertemplate='<b>%{text}</b><br>Temperature: %{x}<br>Top-p: %{y}<br>점수: %{marker.color:.3f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=2
        )

        # 레이아웃 업데이트
        fig.update_layout(
            title='LLM 최적화 결과 종합 분석',
            height=800,
            showlegend=False
        )

        # 축 레이블 설정
        fig.update_xaxes(title_text="모델", row=1, col=1)
        fig.update_yaxes(title_text="점수", row=1, col=1)

        fig.update_xaxes(title_text="최적화 순서", row=1, col=2)
        fig.update_yaxes(title_text="점수", row=1, col=2)

        fig.update_xaxes(title_text="Temperature", row=2, col=2)
        fig.update_yaxes(title_text="Top-p", row=2, col=2)

        if save_path:
            fig.write_html(self.viz_dir / f"{save_path}.html")
            try:
                fig.write_image(self.viz_dir / f"{save_path}.png")
            except:
                pass  # kaleido가 없어도 HTML은 저장됨

        return fig

    def create_benchmark_analysis(self, results: List[Dict[str, Any]],
                                save_path: str = None) -> go.Figure:
        """벤치마크 결과 분석 차트 생성"""
        if not results:
            return None

        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '모델별 처리 속도',
                '지연시간 분포',
                '메모리 사용량 vs 성능',
                '비용 효율성'
            ),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # 데이터 준비
        models = []
        tokens_per_sec = []
        latencies = []
        memory_usage = []
        costs = []

        for result in results:
            model = result.get('model_name', 'Unknown')
            perf = result.get('performance_metrics', {})
            cost = result.get('cost_analysis', {})

            models.append(model)
            tokens_per_sec.append(perf.get('tokens_per_second', 0))
            latencies.append([
                perf.get('latency_p50', 0),
                perf.get('latency_p95', 0),
                perf.get('latency_p99', 0)
            ])
            memory_usage.append(perf.get('memory_usage_mb', 0))
            costs.append(cost.get('cost_per_1k_tokens_usd', 0))

        # 1. 처리 속도 막대 차트
        fig.add_trace(
            go.Bar(
                x=models,
                y=tokens_per_sec,
                name='토큰/초',
                marker_color='lightgreen',
                showlegend=False
            ),
            row=1, col=1
        )

        # 2. 지연시간 박스 플롯
        for i, model in enumerate(models):
            fig.add_trace(
                go.Box(
                    y=latencies[i],
                    name=model,
                    showlegend=False
                ),
                row=1, col=2
            )

        # 3. 메모리 사용량 vs 성능 산점도
        fig.add_trace(
            go.Scatter(
                x=memory_usage,
                y=tokens_per_sec,
                mode='markers+text',
                text=models,
                textposition='top center',
                marker=dict(size=12, color='orange'),
                name='메모리 효율성',
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. 비용 효율성
        if any(costs):
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=costs,
                    name='1K토큰당 비용($)',
                    marker_color='red',
                    showlegend=False
                ),
                row=2, col=2
            )

        # 레이아웃 업데이트
        fig.update_layout(
            title='LLM 벤치마크 결과 분석',
            height=800
        )

        # 축 레이블 설정
        fig.update_xaxes(title_text="모델", row=1, col=1)
        fig.update_yaxes(title_text="토큰/초", row=1, col=1)

        fig.update_yaxes(title_text="지연시간 (초)", row=1, col=2)

        fig.update_xaxes(title_text="메모리 사용량 (MB)", row=2, col=1)
        fig.update_yaxes(title_text="토큰/초", row=2, col=1)

        fig.update_xaxes(title_text="모델", row=2, col=2)
        fig.update_yaxes(title_text="비용 ($)", row=2, col=2)

        if save_path:
            fig.write_html(self.viz_dir / f"{save_path}.html")
            try:
                fig.write_image(self.viz_dir / f"{save_path}.png")
            except:
                pass

        return fig

    def create_model_comparison_chart(self, results: List[Dict[str, Any]],
                                    save_path: str = None) -> go.Figure:
        """모델 비교 차트 생성"""
        if not results:
            return None

        # 모델별 최고 성능 집계
        model_data = {}

        for result in results:
            model = result.get('model_name', 'Unknown')

            if model not in model_data:
                model_data[model] = {
                    'best_score': 0,
                    'best_speed': 0,
                    'min_memory': float('inf'),
                    'min_cost': float('inf'),
                    'datasets': set()
                }

            # 최적화 결과
            if 'best_score' in result:
                score = result['best_score']
                if score > model_data[model]['best_score']:
                    model_data[model]['best_score'] = score

            # 벤치마크 결과
            if 'performance_metrics' in result:
                perf = result['performance_metrics']
                speed = perf.get('tokens_per_second', 0)
                memory = perf.get('memory_usage_mb', float('inf'))

                if speed > model_data[model]['best_speed']:
                    model_data[model]['best_speed'] = speed

                if memory < model_data[model]['min_memory']:
                    model_data[model]['min_memory'] = memory

            # 비용 정보
            if 'cost_analysis' in result:
                cost = result['cost_analysis'].get('cost_per_1k_tokens_usd', float('inf'))
                if cost < model_data[model]['min_cost']:
                    model_data[model]['min_cost'] = cost

            # 데이터셋 정보
            dataset = result.get('dataset_name', '')
            if dataset:
                model_data[model]['datasets'].add(dataset)

        # 레이더 차트 생성
        fig = go.Figure()

        categories = ['정확도', '속도', '메모리 효율성', '비용 효율성', '범용성']

        for model, data in model_data.items():
            # 정규화된 점수 계산
            accuracy_score = data['best_score']
            speed_score = min(data['best_speed'] / 100, 1.0)  # 100 tokens/sec = 1.0
            memory_score = max(0, 1.0 - (data['min_memory'] / 10000))  # 10GB = 0점
            cost_score = max(0, 1.0 - (data['min_cost'] * 10000))  # 낮은 비용이 높은 점수
            versatility_score = min(len(data['datasets']) / 5, 1.0)  # 5개 데이터셋 = 1.0

            values = [accuracy_score, speed_score, memory_score, cost_score, versatility_score]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=model,
                line=dict(width=2)
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='모델별 종합 성능 비교',
            showlegend=True,
            height=600
        )

        if save_path:
            fig.write_html(self.viz_dir / f"{save_path}.html")
            try:
                fig.write_image(self.viz_dir / f"{save_path}.png")
            except:
                pass

        return fig

    def create_hardware_analysis(self, hardware_info: Dict[str, Any],
                               save_path: str = None) -> go.Figure:
        """하드웨어 분석 차트 생성"""

        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'GPU 메모리 현황',
                '시스템 메모리 사용률',
                '추천 모델 크기',
                '성능 예측'
            ),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        # 1. GPU 메모리 현황
        if hardware_info.get('cuda_available', False):
            gpu_names = []
            gpu_memories = []

            for i in range(hardware_info['cuda_device_count']):
                gpu_name = hardware_info.get(f'gpu_{i}_name', f'GPU {i}')
                gpu_memory = hardware_info.get(f'gpu_{i}_memory', 0)
                gpu_names.append(gpu_name)
                gpu_memories.append(gpu_memory)

            fig.add_trace(
                go.Bar(
                    x=gpu_names,
                    y=gpu_memories,
                    name='GPU 메모리 (GB)',
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=1, col=1
            )

        # 2. 시스템 메모리 사용률
        total_memory = hardware_info.get('total_memory', 0)
        available_memory = hardware_info.get('available_memory', 0)
        used_memory = total_memory - available_memory

        fig.add_trace(
            go.Pie(
                labels=['사용 중', '사용 가능'],
                values=[used_memory, available_memory],
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. 추천 모델 크기
        total_gpu_memory = sum(
            hardware_info.get(f'gpu_{i}_memory', 0)
            for i in range(hardware_info.get('cuda_device_count', 0))
        )

        model_recommendations = self._get_model_recommendations(total_gpu_memory)

        fig.add_trace(
            go.Bar(
                x=list(model_recommendations.keys()),
                y=list(model_recommendations.values()),
                name='추천 점수',
                marker_color='lightgreen',
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. 성능 예측
        model_sizes = ['7B', '13B', '30B', '70B']
        predicted_speeds = self._predict_performance(hardware_info, model_sizes)

        fig.add_trace(
            go.Scatter(
                x=model_sizes,
                y=predicted_speeds,
                mode='lines+markers',
                name='예상 속도',
                line=dict(color='orange', width=3),
                marker=dict(size=8),
                showlegend=False
            ),
            row=2, col=2
        )

        # 레이아웃 업데이트
        fig.update_layout(
            title='하드웨어 분석 및 성능 예측',
            height=800
        )

        # 축 레이블 설정
        fig.update_xaxes(title_text="GPU", row=1, col=1)
        fig.update_yaxes(title_text="메모리 (GB)", row=1, col=1)

        fig.update_xaxes(title_text="모델 크기", row=2, col=1)
        fig.update_yaxes(title_text="추천 점수", row=2, col=1)

        fig.update_xaxes(title_text="모델 크기", row=2, col=2)
        fig.update_yaxes(title_text="예상 속도 (tokens/sec)", row=2, col=2)

        if save_path:
            fig.write_html(self.viz_dir / f"{save_path}.html")
            try:
                fig.write_image(self.viz_dir / f"{save_path}.png")
            except:
                pass

        return fig

    def _get_model_recommendations(self, total_gpu_memory: float) -> Dict[str, float]:
        """GPU 메모리에 따른 모델 추천 점수"""
        recommendations = {}

        if total_gpu_memory >= 160:
            recommendations.update({'70B': 1.0, '30B': 1.0, '13B': 1.0, '7B': 1.0})
        elif total_gpu_memory >= 80:
            recommendations.update({'70B': 0.7, '30B': 1.0, '13B': 1.0, '7B': 1.0})
        elif total_gpu_memory >= 32:
            recommendations.update({'70B': 0.3, '30B': 0.8, '13B': 1.0, '7B': 1.0})
        elif total_gpu_memory >= 16:
            recommendations.update({'70B': 0.0, '30B': 0.2, '13B': 0.7, '7B': 1.0})
        elif total_gpu_memory >= 8:
            recommendations.update({'70B': 0.0, '30B': 0.0, '13B': 0.3, '7B': 0.8})
        else:
            recommendations.update({'70B': 0.0, '30B': 0.0, '13B': 0.0, '7B': 0.5})

        return recommendations

    def _predict_performance(self, hardware_info: Dict[str, Any], model_sizes: List[str]) -> List[float]:
        """모델 크기별 성능 예측"""
        total_gpu_memory = sum(
            hardware_info.get(f'gpu_{i}_memory', 0)
            for i in range(hardware_info.get('cuda_device_count', 0))
        )

        # 간단한 성능 예측 모델 (실제로는 더 복잡한 모델 사용)
        base_performance = min(total_gpu_memory * 5, 100)  # GPU 메모리 기반 기본 성능

        predictions = []
        for size in model_sizes:
            if size == '7B':
                pred = base_performance * 0.9
            elif size == '13B':
                pred = base_performance * 0.7
            elif size == '30B':
                pred = base_performance * 0.4
            elif size == '70B':
                pred = base_performance * 0.2
            else:
                pred = base_performance * 0.5

            predictions.append(max(pred, 0))

        return predictions

    def create_comprehensive_dashboard(self, save_path: str = "dashboard") -> go.Figure:
        """종합 대시보드 생성"""
        # 모든 결과 로드
        optimization_results = self.load_optimization_results()
        benchmark_results = self.load_benchmark_results()

        # 큰 서브플롯 생성
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                '최적화 성능 트렌드', '모델별 처리 속도', '파라미터 분포',
                '메모리 사용량', '비용 분석', '정확도 vs 속도',
                '하드웨어 효율성', '데이터셋별 성능', '종합 점수'
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "histogram"}],
                   [{"type": "box"}, {"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}, {"type": "polar"}]]
        )

        # 각 차트에 데이터 추가 (간소화)
        if optimization_results:
            # 1. 최적화 트렌드
            scores = [r.get('best_score', 0) for r in optimization_results]
            fig.add_trace(
                go.Scatter(y=scores, mode='lines+markers', name='최적화 점수', showlegend=False),
                row=1, col=1
            )

        if benchmark_results:
            # 2. 모델별 속도
            models = [r.get('model_name', 'Unknown') for r in benchmark_results]
            speeds = [r.get('performance_metrics', {}).get('tokens_per_second', 0) for r in benchmark_results]
            fig.add_trace(
                go.Bar(x=models, y=speeds, name='처리 속도', showlegend=False),
                row=1, col=2
            )

        # 추가 차트들도 유사하게 구현...

        fig.update_layout(
            title='LLM 최적화 종합 대시보드',
            height=1200,
            showlegend=False
        )

        if save_path:
            fig.write_html(self.viz_dir / f"{save_path}.html")
            try:
                fig.write_image(self.viz_dir / f"{save_path}.png")
            except:
                pass

        return fig

    def generate_performance_report(self, save_path: str = "performance_report.html"):
        """종합 성능 리포트 생성"""
        optimization_results = self.load_optimization_results()
        benchmark_results = self.load_benchmark_results()

        # HTML 리포트 생성
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>오픈소스 LLM 성능 최적화 리포트</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: linear-gradient(90deg, #4CAF50, #45a049); color: white; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #4CAF50; background-color: #f9f9f9; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4CAF50; color: white; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                         background: linear-gradient(135deg, #e3f2fd, #bbdefb); border-radius: 8px; 
                         box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .chart-container {{ margin: 20px 0; text-align: center; }}
                .highlight {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .warning {{ background-color: #f8d7da; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .success {{ background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
        """

        # 헤더
        html_content += f"""
        <div class="header">
            <h1>🚀 오픈소스 LLM 성능 최적화 리포트</h1>
            <p>생성 일시: {datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")}</p>
            <p>총 최적화 실험: {len(optimization_results)}개 | 벤치마크 테스트: {len(benchmark_results)}개</p>
        </div>
        """

        # 요약 통계
        if optimization_results or benchmark_results:
            html_content += """
            <div class="section">
                <h2>📊 성능 요약</h2>
            """

            if optimization_results:
                best_opt = max(optimization_results, key=lambda x: x.get('best_score', 0))
                avg_score = np.mean([r.get('best_score', 0) for r in optimization_results])

                html_content += f"""
                <div class="metric">
                    <strong>최고 최적화 점수</strong><br>
                    {best_opt.get('best_score', 0):.3f}<br>
                    <small>모델: {best_opt.get('model_name', 'Unknown')}</small>
                </div>
                <div class="metric">
                    <strong>평균 최적화 점수</strong><br>
                    {avg_score:.3f}<br>
                    <small>전체 {len(optimization_results)}개 실험</small>
                </div>
                """

            if benchmark_results:
                fastest = max(benchmark_results,
                            key=lambda x: x.get('performance_metrics', {}).get('tokens_per_second', 0))
                avg_speed = np.mean([r.get('performance_metrics', {}).get('tokens_per_second', 0)
                                   for r in benchmark_results])

                html_content += f"""
                <div class="metric">
                    <strong>최고 처리 속도</strong><br>
                    {fastest.get('performance_metrics', {}).get('tokens_per_second', 0):.1f} tokens/sec<br>
                    <small>모델: {fastest.get('model_name', 'Unknown')}</small>
                </div>
                <div class="metric">
                    <strong>평균 처리 속도</strong><br>
                    {avg_speed:.1f} tokens/sec<br>
                    <small>전체 {len(benchmark_results)}개 테스트</small>
                </div>
                """

            html_content += "</div>"

        # 모델별 상세 분석
        if optimization_results:
            html_content += """
            <div class="section">
                <h2>🔧 최적화 결과 상세 분석</h2>
                <table>
                    <tr>
                        <th>모델</th>
                        <th>데이터셋</th>
                        <th>최고 점수</th>
                        <th>최적 Temperature</th>
                        <th>최적 Top-p</th>
                        <th>소요 시간</th>
                    </tr>
            """

            for result in sorted(optimization_results, key=lambda x: x.get('best_score', 0), reverse=True):
                best_params = result.get('best_params', {})
                html_content += f"""
                <tr>
                    <td>{result.get('model_name', 'Unknown')}</td>
                    <td>{result.get('dataset_name', 'Unknown')}</td>
                    <td>{result.get('best_score', 0):.3f}</td>
                    <td>{best_params.get('temperature', 0):.3f}</td>
                    <td>{best_params.get('top_p', 0):.3f}</td>
                    <td>{result.get('total_time', 0):.1f}초</td>
                </tr>
                """

            html_content += "</table></div>"

        # 벤치마크 결과
        if benchmark_results:
            html_content += """
            <div class="section">
                <h2>⚡ 벤치마크 결과 분석</h2>
                <table>
                    <tr>
                        <th>모델</th>
                        <th>처리 속도</th>
                        <th>지연시간 P95</th>
                        <th>메모리 사용량</th>
                        <th>1K토큰당 비용</th>
                        <th>효율성 점수</th>
                    </tr>
            """

            for result in sorted(benchmark_results,
                               key=lambda x: x.get('performance_metrics', {}).get('tokens_per_second', 0),
                               reverse=True):
                perf = result.get('performance_metrics', {})
                cost = result.get('cost_analysis', {})
                eff = result.get('hardware_efficiency', {})

                html_content += f"""
                <tr>
                    <td>{result.get('model_name', 'Unknown')}</td>
                    <td>{perf.get('tokens_per_second', 0):.1f} tokens/sec</td>
                    <td>{perf.get('latency_p95', 0):.3f}초</td>
                    <td>{perf.get('memory_usage_mb', 0):.0f}MB</td>
                    <td>${cost.get('cost_per_1k_tokens_usd', 0):.6f}</td>
                    <td>{eff.get('overall_efficiency', 0):.3f}</td>
                </tr>
                """

            html_content += "</table></div>"

        # 최적 설정 추천
        html_content += """
        <div class="section">
            <h2>💡 최적 설정 추천</h2>
        """

        if optimization_results:
            # 최고 성능 모델
            best_model = max(optimization_results, key=lambda x: x.get('best_score', 0))
            best_params = best_model.get('best_params', {})

            html_content += f"""
            <div class="success">
                <h3>🏆 최고 성능 설정</h3>
                <p><strong>모델:</strong> {best_model.get('model_name', 'Unknown')}</p>
                <p><strong>데이터셋:</strong> {best_model.get('dataset_name', 'Unknown')}</p>
                <p><strong>추천 파라미터:</strong></p>
                <ul>
                    <li>Temperature: {best_params.get('temperature', 0):.3f}</li>
                    <li>Top-p: {best_params.get('top_p', 0):.3f}</li>
                    <li>Top-k: {best_params.get('top_k', 0)}</li>
                    <li>Max tokens: {best_params.get('max_new_tokens', 0)}</li>
                    <li>Repetition penalty: {best_params.get('repetition_penalty', 0):.3f}</li>
                </ul>
            </div>
            """

        if benchmark_results:
            # 최고 효율성 모델
            fastest_model = max(benchmark_results,
                              key=lambda x: x.get('performance_metrics', {}).get('tokens_per_second', 0))

            html_content += f"""
            <div class="highlight">
                <h3>⚡ 최고 속도 모델</h3>
                <p><strong>모델:</strong> {fastest_model.get('model_name', 'Unknown')}</p>
                <p><strong>처리 속도:</strong> {fastest_model.get('performance_metrics', {}).get('tokens_per_second', 0):.1f} tokens/sec</p>
                <p><strong>추천 용도:</strong> 대용량 배치 처리, 실시간 응답</p>
            </div>
            """

        # 일반적인 추천사항
        html_content += """
        <div class="section">
            <h3>🎯 일반 추천사항</h3>
            <ul>
                <li><strong>정확도 우선:</strong> Temperature 0.0-0.3, Top-p 0.1-0.5 사용</li>
                <li><strong>창의성 우선:</strong> Temperature 0.7-1.0, Top-p 0.8-0.95 사용</li>
                <li><strong>메모리 최적화:</strong> 4-bit 양자화 활용, 배치 크기 조절</li>
                <li><strong>속도 최적화:</strong> vLLM 엔진 사용, Flash Attention 활용</li>
                <li><strong>비용 효율성:</strong> 모델 크기와 성능의 균형점 찾기</li>
            </ul>
        </div>
        """

        # 하드웨어 최적화 팁
        html_content += """
        <div class="section">
            <h2>💻 하드웨어 최적화 팁</h2>
            <div class="warning">
                <h3>⚠️ 주의사항</h3>
                <ul>
                    <li>GPU 메모리 사용률이 90%를 넘지 않도록 주의</li>
                    <li>대용량 모델은 텐서 병렬화 고려</li>
                    <li>배치 크기는 메모리 한계 내에서 최대한 크게 설정</li>
                </ul>
            </div>
            
            <h3>📈 성능 향상 방법</h3>
            <ul>
                <li><strong>양자화:</strong> 4-bit 양자화로 메모리 사용량 75% 감소</li>
                <li><strong>컴파일 최적화:</strong> torch.compile() 사용으로 10-20% 속도 향상</li>
                <li><strong>동적 배칭:</strong> vLLM의 동적 배칭으로 처리량 증대</li>
                <li><strong>Flash Attention:</strong> 메모리 효율적인 어텐션 메커니즘</li>
            </ul>
        </div>
        """

        # 마무리
        html_content += """
        <div class="section">
            <h2>📈 결론 및 다음 단계</h2>
            <p>이 리포트는 오픈소스 LLM의 추론 성능 최적화 결과를 종합적으로 분석한 것입니다.</p>
            
            <h3>다음 단계 제안:</h3>
            <ol>
                <li>최적화된 파라미터를 프로덕션 환경에 적용</li>
                <li>정기적인 성능 모니터링 및 벤치마크 실행</li>
                <li>새로운 모델이나 데이터셋에 대한 추가 최적화</li>
                <li>하드웨어 업그레이드 시 재최적화 수행</li>
            </ol>
            
            <p><small>리포트 생성 시스템: 오픈소스 LLM 추론 성능 최적화 도구</small></p>
        </div>
        
        </body>
        </html>
        """

        # 파일 저장
        report_path = self.viz_dir / save_path
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"성능 리포트 생성 완료: {report_path}")
        return report_path

class PerformanceAnalyzer:
    """성능 분석 유틸리티 클래스"""

    @staticmethod
    def analyze_optimization_trends(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """최적화 트렌드 분석"""
        if not results:
            return {}

        # 시간순 정렬
        sorted_results = sorted(results, key=lambda x: x.get('timestamp', ''))

        scores = [r.get('best_score', 0) for r in sorted_results]

        analysis = {
            'total_experiments': len(results),
            'score_trend': 'improving' if scores[-1] > scores[0] else 'declining',
            'best_score': max(scores),
            'worst_score': min(scores),
            'average_score': np.mean(scores),
            'score_variance': np.var(scores),
            'improvement_rate': (scores[-1] - scores[0]) / len(scores) if len(scores) > 1 else 0
        }

        return analysis

    @staticmethod
    def find_optimal_parameters(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """최적 파라미터 패턴 찾기"""
        if not results:
            return {}

        # 파라미터별 성능 분석
        param_performance = {
            'temperature': {},
            'top_p': {},
            'top_k': {},
            'max_new_tokens': {},
            'repetition_penalty': {}
        }

        for result in results:
            score = result.get('best_score', 0)
            params = result.get('best_params', {})

            for param_name in param_performance.keys():
                param_value = params.get(param_name, 0)

                # 값을 구간으로 그룹화
                if param_name == 'temperature':
                    bucket = round(param_value, 1)
                elif param_name == 'top_p':
                    bucket = round(param_value, 1)
                elif param_name == 'top_k':
                    bucket = int(param_value // 10) * 10  # 10단위로 그룹화
                elif param_name == 'max_new_tokens':
                    bucket = int(param_value // 100) * 100  # 100단위로 그룹화
                elif param_name == 'repetition_penalty':
                    bucket = round(param_value, 2)

                if bucket not in param_performance[param_name]:
                    param_performance[param_name][bucket] = []
                param_performance[param_name][bucket].append(score)

        # 각 파라미터의 최적값 찾기
        optimal_params = {}
        for param_name, buckets in param_performance.items():
            if buckets:
                # 각 구간의 평균 성능 계산
                bucket_averages = {
                    bucket: np.mean(scores)
                    for bucket, scores in buckets.items()
                }

                # 최고 성능 구간 찾기
                best_bucket = max(bucket_averages, key=bucket_averages.get)
                optimal_params[param_name] = {
                    'value': best_bucket,
                    'performance': bucket_averages[best_bucket],
                    'sample_count': len(buckets[best_bucket])
                }

        return optimal_params

    @staticmethod
    def calculate_efficiency_score(benchmark_result: Dict[str, Any]) -> float:
        """종합 효율성 점수 계산"""
        perf = benchmark_result.get('performance_metrics', {})
        cost = benchmark_result.get('cost_analysis', {})

        # 정규화된 점수들
        speed_score = min(perf.get('tokens_per_second', 0) / 100, 1.0)
        memory_score = max(0, 1.0 - (perf.get('memory_usage_mb', 0) / 10000))
        cost_score = max(0, 1.0 - (cost.get('cost_per_1k_tokens_usd', 0) * 10000))
        latency_score = max(0, 1.0 - perf.get('latency_p95', 0))

        # 가중 평균
        efficiency_score = (
            speed_score * 0.3 +
            memory_score * 0.25 +
            cost_score * 0.25 +
            latency_score * 0.2
        )

        return efficiency_score

# 사용 예시
if __name__ == "__main__":
    # 시각화 객체 생성
    visualizer = ResultVisualizer()

    print("=== 오픈소스 LLM 최적화 시각화 시스템 ===")

    # 더미 최적화 결과 생성 (테스트용)
    dummy_optimization = [
        {
            'model_name': 'llama2-7b',
            'dataset_name': 'korean_math',
            'best_score': 0.85,
            'best_params': {
                'temperature': 0.1,
                'top_p': 0.3,
                'top_k': 10,
                'max_new_tokens': 200,
                'repetition_penalty': 1.05
            },
            'total_time': 1200,
            'hardware_usage': {
                'cpu_avg': 45.2,
                'memory_peak': 78.5,
                'gpu_memory_peak': 12.3
            },
            'timestamp': '2024-01-15T10:30:00'
        },
        {
            'model_name': 'mistral-7b',
            'dataset_name': 'korean_qa',
            'best_score': 0.82,
            'best_params': {
                'temperature': 0.2,
                'top_p': 0.5,
                'top_k': 20,
                'max_new_tokens': 300,
                'repetition_penalty': 1.1
            },
            'total_time': 980,
            'hardware_usage': {
                'cpu_avg': 52.1,
                'memory_peak': 82.3,
                'gpu_memory_peak': 10.8
            },
            'timestamp': '2024-01-15T11:45:00'
        }
    ]

    # 더미 벤치마크 결과 생성
    dummy_benchmark = [
        {
            'model_name': 'llama2-7b',
            'dataset_name': 'korean_math',
            'performance_metrics': {
                'tokens_per_second': 75.5,
                'latency_p50': 0.8,
                'latency_p95': 1.2,
                'latency_p99': 1.8,
                'memory_usage_mb': 6400,
                'throughput': 1.25
            },
            'cost_analysis': {
                'cost_per_hour_usd': 0.15,
                'cost_per_1k_tokens_usd': 0.000025
            },
            'hardware_efficiency': {
                'tokens_per_watt': 0.755,
                'memory_efficiency': 0.0118,
                'overall_efficiency': 0.342
            },
            'timestamp': '2024-01-15T12:00:00'
        }
    ]

    try:
        # 1. 최적화 분석 차트
        if dummy_optimization:
            print("1. 최적화 분석 차트 생성...")
            fig1 = visualizer.create_optimization_analysis(dummy_optimization, "test_optimization")
            print("   ✅ 완료: test_optimization.html")

        # 2. 벤치마크 분석 차트
        if dummy_benchmark:
            print("2. 벤치마크 분석 차트 생성...")
            fig2 = visualizer.create_benchmark_analysis(dummy_benchmark, "test_benchmark")
            print("   ✅ 완료: test_benchmark.html")

        # 3. 모델 비교 차트
        print("3. 모델 비교 차트 생성...")
        all_results = dummy_optimization + dummy_benchmark
        fig3 = visualizer.create_model_comparison_chart(all_results, "test_comparison")
        print("   ✅ 완료: test_comparison.html")

        # 4. 하드웨어 분석
        print("4. 하드웨어 분석 차트 생성...")
        from config import HardwareDetector
        hardware_info = HardwareDetector.detect_hardware()
        fig4 = visualizer.create_hardware_analysis(hardware_info, "test_hardware")
        print("   ✅ 완료: test_hardware.html")

        # 5. 종합 대시보드
        print("5. 종합 대시보드 생성...")
        fig5 = visualizer.create_comprehensive_dashboard("test_dashboard")
        print("   ✅ 완료: test_dashboard.html")

        # 6. 성능 리포트
        print("6. 성능 리포트 생성...")
        report_path = visualizer.generate_performance_report("test_report.html")
        print(f"   ✅ 완료: {report_path}")

        # 7. 성능 분석
        print("7. 성능 분석 실행...")
        analyzer = PerformanceAnalyzer()

        trends = analyzer.analyze_optimization_trends(dummy_optimization)
        print(f"   최적화 트렌드: {trends.get('score_trend', 'unknown')}")
        print(f"   평균 점수: {trends.get('average_score', 0):.3f}")

        optimal_params = analyzer.find_optimal_parameters(dummy_optimization)
        print(f"   최적 Temperature: {optimal_params.get('temperature', {}).get('value', 'N/A')}")

        if dummy_benchmark:
            efficiency = analyzer.calculate_efficiency_score(dummy_benchmark[0])
            print(f"   효율성 점수: {efficiency:.3f}")

        print(f"\n📁 모든 시각화 파일이 'visualizations' 디렉토리에 저장되었습니다.")
        print(f"🌐 HTML 파일을 브라우저에서 열어 인터랙티브 차트를 확인하세요.")

    except Exception as e:
        print(f"❌ 시각화 생성 실패: {e}")
        import traceback
        traceback.print_exc()