#!/usr/bin/env python3
"""
Cost Analysis Module for Phase 5

This module provides detailed cost-effectiveness analysis comparing human labor costs
with LLM API costs across different scenarios and use cases.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class CostScenario:
    """Configuration for different cost analysis scenarios"""
    name: str
    human_hourly_rate: float
    human_overhead_multiplier: float = 1.5  # Benefits, equipment, etc.
    human_training_hours: float = 8.0  # Initial training time
    human_training_cost_per_hour: float = 50.0  # Trainer cost
    llm_setup_cost: float = 1000.0  # One-time setup cost
    llm_monthly_subscription: float = 0.0  # Monthly subscription if applicable
    tasks_per_month: int = 1000  # Expected task volume

class CostAnalyzer:
    """
    Advanced cost analysis for human vs LLM comparison
    """
    
    def __init__(self):
        self.scenarios = self._create_default_scenarios()
    
    def _create_default_scenarios(self) -> List[CostScenario]:
        """Create default cost scenarios for analysis"""
        return [
            CostScenario(
                name="Manufacturing Technician",
                human_hourly_rate=25.0,
                human_overhead_multiplier=1.5,
                human_training_hours=16.0,
                human_training_cost_per_hour=40.0
            ),
            CostScenario(
                name="Quality Assurance Specialist",
                human_hourly_rate=35.0,
                human_overhead_multiplier=1.6,
                human_training_hours=24.0,
                human_training_cost_per_hour=50.0
            ),
            CostScenario(
                name="Data Analyst",
                human_hourly_rate=45.0,
                human_overhead_multiplier=1.7,
                human_training_hours=32.0,
                human_training_cost_per_hour=60.0
            ),
            CostScenario(
                name="Senior Engineer",
                human_hourly_rate=65.0,
                human_overhead_multiplier=1.8,
                human_training_hours=40.0,
                human_training_cost_per_hour=80.0
            )
        ]
    
    def calculate_total_human_cost_per_task(self, scenario: CostScenario, 
                                          avg_time_seconds: float,
                                          monthly_task_volume: int = None) -> Dict[str, float]:
        """Calculate comprehensive human cost per task including all factors"""
        
        if monthly_task_volume is None:
            monthly_task_volume = scenario.tasks_per_month
        
        # Base labor cost per task
        hours_per_task = avg_time_seconds / 3600
        base_labor_cost = hours_per_task * scenario.human_hourly_rate
        
        # Total labor cost with overhead
        total_labor_cost = base_labor_cost * scenario.human_overhead_multiplier
        
        # Training cost amortized over monthly tasks
        total_training_cost = (scenario.human_training_hours * 
                             scenario.human_training_cost_per_hour)
        training_cost_per_task = total_training_cost / monthly_task_volume
        
        # Total cost per task
        total_cost_per_task = total_labor_cost + training_cost_per_task
        
        return {
            'base_labor_cost': base_labor_cost,
            'overhead_cost': total_labor_cost - base_labor_cost,
            'training_cost_per_task': training_cost_per_task,
            'total_cost_per_task': total_cost_per_task,
            'effective_hourly_rate': total_cost_per_task / hours_per_task
        }
    
    def calculate_llm_cost_with_setup(self, avg_api_cost: float,
                                    monthly_task_volume: int,
                                    setup_cost: float = 1000.0,
                                    monthly_subscription: float = 0.0) -> Dict[str, float]:
        """Calculate LLM cost including setup and subscription costs"""
        
        # Setup cost amortized over 12 months
        setup_cost_per_task = setup_cost / (12 * monthly_task_volume)
        
        # Monthly subscription cost per task
        subscription_cost_per_task = monthly_subscription / monthly_task_volume
        
        # Total cost per task
        total_cost_per_task = avg_api_cost + setup_cost_per_task + subscription_cost_per_task
        
        return {
            'api_cost_per_task': avg_api_cost,
            'setup_cost_per_task': setup_cost_per_task,
            'subscription_cost_per_task': subscription_cost_per_task,
            'total_cost_per_task': total_cost_per_task
        }
    
    def perform_roi_analysis(self, human_accuracy: float, llm_accuracy: float,
                           human_cost_per_task: float, llm_cost_per_task: float,
                           error_cost: float = 100.0) -> Dict[str, float]:
        """Perform ROI analysis considering accuracy and error costs"""
        
        # Cost of errors
        human_error_cost = (1 - human_accuracy) * error_cost
        llm_error_cost = (1 - llm_accuracy) * error_cost
        
        # Total cost including error costs
        human_total_cost = human_cost_per_task + human_error_cost
        llm_total_cost = llm_cost_per_task + llm_error_cost
        
        # ROI calculations
        cost_savings = human_total_cost - llm_total_cost
        roi_percentage = (cost_savings / human_total_cost) * 100 if human_total_cost > 0 else 0
        
        # Break-even analysis
        break_even_tasks = 0
        if cost_savings > 0:
            # Calculate how many tasks needed to break even on setup costs
            setup_cost = 1000.0  # Default setup cost
            break_even_tasks = setup_cost / cost_savings if cost_savings > 0 else float('inf')
        
        return {
            'human_total_cost': human_total_cost,
            'llm_total_cost': llm_total_cost,
            'cost_savings_per_task': cost_savings,
            'roi_percentage': roi_percentage,
            'break_even_tasks': break_even_tasks,
            'human_error_cost': human_error_cost,
            'llm_error_cost': llm_error_cost
        }
    
    def generate_cost_comparison_report(self, human_performance: Dict[str, float],
                                      llm_performance: Dict[str, float],
                                      monthly_volume: int = 1000) -> Dict[str, Any]:
        """Generate comprehensive cost comparison report"""
        
        report = {
            'scenarios': {},
            'summary': {},
            'recommendations': []
        }
        
        # Analyze each scenario
        for scenario in self.scenarios:
            human_costs = self.calculate_total_human_cost_per_task(
                scenario, 
                human_performance['avg_time_sec'],
                monthly_volume
            )
            
            llm_costs = self.calculate_llm_cost_with_setup(
                llm_performance['avg_cost_usd'],
                monthly_volume,
                scenario.llm_setup_cost,
                scenario.llm_monthly_subscription
            )
            
            roi_analysis = self.perform_roi_analysis(
                human_performance['accuracy'],
                llm_performance['accuracy'],
                human_costs['total_cost_per_task'],
                llm_costs['total_cost_per_task']
            )
            
            report['scenarios'][scenario.name] = {
                'human_costs': human_costs,
                'llm_costs': llm_costs,
                'roi_analysis': roi_analysis,
                'scenario_config': scenario
            }
        
        # Generate summary and recommendations
        best_roi_scenario = max(report['scenarios'].items(), 
                              key=lambda x: x[1]['roi_analysis']['roi_percentage'])
        
        report['summary'] = {
            'best_roi_scenario': best_roi_scenario[0],
            'best_roi_percentage': best_roi_scenario[1]['roi_analysis']['roi_percentage'],
            'average_cost_savings': np.mean([s['roi_analysis']['cost_savings_per_task'] 
                                           for s in report['scenarios'].values()]),
            'monthly_volume_analyzed': monthly_volume
        }
        
        # Generate recommendations
        if report['summary']['best_roi_percentage'] > 20:
            report['recommendations'].append("Strong ROI case for LLM implementation")
        elif report['summary']['best_roi_percentage'] > 0:
            report['recommendations'].append("Moderate ROI case for LLM implementation")
        else:
            report['recommendations'].append("Human performance may be more cost-effective")
        
        if llm_performance['accuracy'] > human_performance['accuracy']:
            report['recommendations'].append("LLM shows superior accuracy performance")
        
        return report
    
    def create_cost_visualization(self, cost_report: Dict[str, Any], 
                                output_path: str = None) -> str:
        """Create comprehensive cost analysis visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cost-Effectiveness Analysis: Human vs LLM', fontsize=16, fontweight='bold')
        
        scenarios = list(cost_report['scenarios'].keys())
        
        # 1. Total cost per task comparison
        human_costs = [cost_report['scenarios'][s]['human_costs']['total_cost_per_task'] 
                      for s in scenarios]
        llm_costs = [cost_report['scenarios'][s]['llm_costs']['total_cost_per_task'] 
                    for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, human_costs, width, label='Human', color='#FF6B6B', alpha=0.8)
        axes[0, 0].bar(x + width/2, llm_costs, width, label='LLM', color='#4ECDC4', alpha=0.8)
        axes[0, 0].set_title('Total Cost per Task by Scenario')
        axes[0, 0].set_ylabel('Cost (USD)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([s.replace(' ', '***REMOVED***n') for s in scenarios], fontsize=9)
        axes[0, 0].legend()
        
        # 2. ROI percentage
        roi_percentages = [cost_report['scenarios'][s]['roi_analysis']['roi_percentage'] 
                          for s in scenarios]
        
        colors = ['green' if roi > 0 else 'red' for roi in roi_percentages]
        axes[0, 1].bar(scenarios, roi_percentages, color=colors, alpha=0.7)
        axes[0, 1].set_title('ROI Percentage by Scenario')
        axes[0, 1].set_ylabel('ROI (%)')
        axes[0, 1].set_xticklabels([s.replace(' ', '***REMOVED***n') for s in scenarios], fontsize=9)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. Break-even analysis
        break_even_tasks = [cost_report['scenarios'][s]['roi_analysis']['break_even_tasks'] 
                           for s in scenarios]
        # Cap at reasonable maximum for visualization
        break_even_capped = [min(be, 10000) if be != float('inf') else 10000 for be in break_even_tasks]
        
        axes[1, 0].bar(scenarios, break_even_capped, color='#95E1D3', alpha=0.8)
        axes[1, 0].set_title('Break-even Point (Tasks)')
        axes[1, 0].set_ylabel('Number of Tasks')
        axes[1, 0].set_xticklabels([s.replace(' ', '***REMOVED***n') for s in scenarios], fontsize=9)
        
        # 4. Cost breakdown for best scenario
        best_scenario = cost_report['summary']['best_roi_scenario']
        best_data = cost_report['scenarios'][best_scenario]
        
        human_breakdown = [
            best_data['human_costs']['base_labor_cost'],
            best_data['human_costs']['overhead_cost'],
            best_data['human_costs']['training_cost_per_task']
        ]
        
        llm_breakdown = [
            best_data['llm_costs']['api_cost_per_task'],
            best_data['llm_costs']['setup_cost_per_task'],
            best_data['llm_costs']['subscription_cost_per_task']
        ]
        
        breakdown_labels = ['Base/API', 'Overhead/Setup', 'Training/Subscription']
        x_breakdown = np.arange(len(breakdown_labels))
        
        axes[1, 1].bar(x_breakdown - width/2, human_breakdown, width, label='Human', color='#FF6B6B', alpha=0.8)
        axes[1, 1].bar(x_breakdown + width/2, llm_breakdown, width, label='LLM', color='#4ECDC4', alpha=0.8)
        axes[1, 1].set_title(f'Cost Breakdown: {best_scenario}')
        axes[1, 1].set_ylabel('Cost (USD)')
        axes[1, 1].set_xticks(x_breakdown)
        axes[1, 1].set_xticklabels(breakdown_labels)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            plt.show()
            return "displayed"
