import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
import json
warnings.filterwarnings('ignore')

class AdvancedAtmosphericReentrySystem:
    def __init__(self):
        # Constantes físicas
        self.R_earth = 6371e3  # Radio de la Tierra (m)
        self.g0 = 9.81  # Gravedad en superficie (m/s²)
        self.rho0 = 1.225  # Densidad del aire al nivel del mar (kg/m³)
        self.H = 7500  # Altura de escala atmosférica (m)
        
        # Parámetros de la cápsula
        self.CD_base = 1.28  # Coeficiente de arrastre base
        self.A = 4.9  # Área frontal (m²)
        self.m = 6000  # Masa de la cápsula (kg)
        self.L_D_max = 0.25  # Relación sustentación/arrastre máxima
        
        # Parámetros térmicos mejorados
        self.k_thermal = 0.12  # Conductividad térmica del escudo (W/m·K)
        self.cp = 1000  # Capacidad calorífica específica (J/kg·K)
        self.thickness = 0.05  # Espesor del escudo térmico (m)
        self.emissivity = 0.85  # Emisividad del material
        self.stefan_boltzmann = 5.67e-8  # Constante de Stefan-Boltzmann
        self.thermal_soak_factor = 0.15  # Factor de absorción térmica
        self.max_internal_temp = 373  # Temperatura máxima interna permitida (K)
        
        # Sistema de control
        self.control_active = True
        self.target_g_force = 4.0  # Fuerza G objetivo
        self.target_temp = 1800  # Temperatura objetivo máxima (K)
        self.sensor_noise = True  # Habilitar ruido de sensores
        
        # Variables de estado expandidas
        self.telemetry = {
            'time': [], 'altitude': [], 'velocity': [], 'temperature': [],
            'heat_flux': [], 'g_force': [], 'gamma': [], 'distance': [],
            'bank_angle': [], 'control_signal': [], 'thermal_stress': [],
            'ablation_rate': [], 'drag_coefficient': [], 'ml_bank_adjustment': [],
            'blackout': [], 'internal_temp': [], 'mach_number': [],
            'dynamic_pressure': [], 'structural_load': [], 'communication_status': []
        }
        
        # Sistema de Machine Learning
        self.ml_controller = None
        self.scaler = StandardScaler()
        self.training_data = []
        
        # Métricas de performance expandidas
        self.max_heat_flux = 0
        self.max_g_force = 0
        self.max_temp = 0
        self.max_internal_temp = 0
        self.total_ablation = 0
        self.mission_alerts = []
        self.blackout_duration = 0
        self.mission_status = "NOMINAL"
        
        # Estados críticos
        self.thermal_failure = False
        self.structural_failure = False
        self.communication_blackout = False
        
    def atmospheric_density(self, h):
        """Modelo de densidad atmosférica exponencial con capas mejorado"""
        if h < 0:
            return self.rho0
        elif h < 11000:
            # Troposfera
            T = 288.15 - 0.0065 * h
            return 1.225 * (T/288.15)**4.256
        elif h < 25000:
            # Estratosfera inferior
            return 0.3639 * np.exp(-(h-11000)/6341.6)
        elif h < 50000:
            # Estratosfera media
            return 0.0889 * np.exp(-(h-25000)/7023.3)
        else:
            # Estratosfera superior y más allá
            return self.rho0 * np.exp(-h / self.H)
    
    def gravity(self, h):
        """Gravedad variable con la altura"""
        return self.g0 * (self.R_earth / (self.R_earth + h))**2
    
    def drag_coefficient(self, mach, angle_of_attack):
        """Coeficiente de arrastre variable con Mach y ángulo de ataque mejorado"""
        # Modelo más realista de CD vs Mach
        if mach < 0.8:
            cd_mach = self.CD_base
        elif mach < 1.2:
            # Región transónica
            cd_mach = self.CD_base * (1 + 0.5 * (mach - 0.8))
        elif mach < 5:
            # Supersónico
            cd_mach = self.CD_base * (1.4 + 0.2 * (mach - 1))
        else:
            # Hipersónico
            cd_mach = self.CD_base * (2.0 - 0.05 * np.log(mach))
        
        # Efecto del ángulo de ataque
        cd_aoa = cd_mach * (1 + 0.15 * np.sin(angle_of_attack)**2)
        return max(cd_aoa, 0.5)
    
    def lift_coefficient(self, mach, bank_angle):
        """Coeficiente de sustentación independiente mejorado"""
        # Modelo desacoplado de sustentación
        cl_base = 0.5 * self.L_D_max
        
        # Efecto del número de Mach
        if mach < 1:
            cl_mach = cl_base
        elif mach < 3:
            cl_mach = cl_base * (1 - 0.1 * (mach - 1))
        else:
            cl_mach = cl_base * 0.8
        
        # Efecto del ángulo de banco
        CL = cl_mach * np.sin(2 * bank_angle)
        return CL
    
    def heat_flux_calculation(self, rho, v, r_nose=0.5):
        """Cálculo avanzado del flujo de calor con correcciones"""
        # Flujo de calor de estancamiento (Fay-Riddell modificado)
        K = 1.742e-4
        
        # Corrección por compresibilidad
        mach = v / 340
        compression_factor = 1 + 0.2 * mach**2 if mach > 1 else 1
        
        q_stag = K * np.sqrt(rho/r_nose) * (v**3.15) * compression_factor
        
        # Factor de distribución para superficie completa
        distribution_factor = 0.7  # Promedio sobre la superficie
        return q_stag * distribution_factor
    
    def thermal_protection_system(self, heat_flux, dt, current_temp=300):
        """Sistema de protección térmica multicapa mejorado"""
        # Temperatura de superficie con thermal soak
        T_surface = (heat_flux / (self.emissivity * self.stefan_boltzmann))**0.25
        
        # Re-radiación térmica
        re_radiation = self.emissivity * self.stefan_boltzmann * current_temp**4
        net_heat_flux = max(0, heat_flux - re_radiation)
        
        # Temperatura interna con conducción multicapa
        thermal_diffusivity = self.k_thermal / (2000 * self.cp)
        penetration_depth = np.sqrt(thermal_diffusivity * dt)
        
        # Modelo de conducción mejorado
        if penetration_depth > self.thickness:
            # Thermal soak completo
            T_internal = T_surface * (1 - self.thermal_soak_factor)
        else:
            # Conducción parcial
            T_internal = current_temp + (T_surface - current_temp) * (penetration_depth / self.thickness)
        
        # Tasa de ablación mejorada (modelo de Charring)
        if T_surface > 1500:  # Temperatura de ablación
            ablation_rate = 1e-6 * (T_surface - 1500)**1.5  # m/s
            # Efecto cooling de la ablación
            cooling_effect = ablation_rate * 2e6  # Calor latente
            T_surface = max(T_surface - cooling_effect / heat_flux * 100, T_surface * 0.9)
        else:
            ablation_rate = 0
        
        return T_surface, T_internal, ablation_rate
    
    def add_sensor_noise(self, value, noise_level):
        """Agregar ruido a las mediciones de sensores"""
        if self.sensor_noise:
            noise = np.random.normal(0, noise_level)
            return value + noise
        return value
    
    def check_blackout_conditions(self, h, v):
        """Verificar condiciones de blackout de comunicación"""
        # Blackout típico en reentrada: 40-80 km, v > 2500 m/s
        altitude_condition = 40e3 < h < 80e3
        velocity_condition = v > 2500
        plasma_density = self.atmospheric_density(h) * v**2 / 1e8
        
        return altitude_condition and velocity_condition and plasma_density > 0.5
    
    def adaptive_controller(self, state, target_params):
        """Sistema de control adaptativo PID + ML mejorado"""
        v, gamma, h, x, bank_angle = state
        target_g, target_temp = target_params
        
        # Agregar ruido a sensores
        v_measured = self.add_sensor_noise(v, 10)
        h_measured = self.add_sensor_noise(h, 100)
        gamma_measured = self.add_sensor_noise(gamma, 0.01)
        
        # Estado actual
        rho = self.atmospheric_density(h_measured)
        mach = v_measured / 340
        current_g = 0.5 * rho * v_measured**2 * self.drag_coefficient(mach, gamma_measured) * self.A / (self.m * self.g0)
        heat_flux = self.heat_flux_calculation(rho, v_measured)
        current_temp, _, _ = self.thermal_protection_system(heat_flux, 1.0)
        
        # Control PID mejorado
        error_g = current_g - target_g
        error_temp = current_temp - target_temp
        
        # ML Control prediction
        ml_control = 0
        if self.ml_controller is not None:
            try:
                # Preparar features para ML
                features = np.array([[
                    v_measured/8000, 
                    h_measured/120000, 
                    gamma_measured/-20, 
                    current_g/10, 
                    heat_flux/1e6
                ]])
                features_scaled = self.scaler.transform(features)
                ml_control = self.ml_controller.predict(features_scaled)[0]
            except:
                ml_control = 0
        
        # Ajuste del ángulo de banco combinando PID + ML
        if self.control_active:
            # Control proporcional-derivativo
            kp_g = 0.1
            kp_temp = 0.0001
            kd = 0.05
            
            # PID adjustment
            pid_adjustment = -kp_g * error_g - kp_temp * error_temp
            
            # ML adjustment (scaled)
            ml_adjustment = ml_control * 0.05
            
            # Combinar controles
            total_adjustment = pid_adjustment + ml_adjustment
            bank_angle = np.clip(bank_angle + total_adjustment, -np.pi/4, np.pi/4)
            
        # Señal de control
        control_signal = np.sqrt(error_g**2 + (error_temp/1000)**2)
        
        return bank_angle, control_signal, ml_control
    
    def check_critical_conditions(self, v, gamma, h, temp, g_force, internal_temp):
        """Verificar condiciones críticas y generar alertas"""
        alerts = []
        
        # Fallo térmico
        if temp > 2500 or internal_temp > self.max_internal_temp:
            alerts.append("🔥 FALLO TÉRMICO: TPS excedido")
            self.thermal_failure = True
            self.mission_status = "CRÍTICO"
        
        # Fuerza G crítica
        if g_force > 10:
            alerts.append("🛑 FUERZA G CRÍTICA: Límite estructural superado")
            self.structural_failure = True
            self.mission_status = "CRÍTICO"
        
        # Velocidad excesiva a baja altitud
        if h < 10000 and v > 500:
            alerts.append("⚠️ VELOCIDAD EXCESIVA: Riesgo de impacto duro")
            self.mission_status = "ALERTA"
        
        # Ángulo de reentrada crítico
        if abs(np.degrees(gamma)) > 20:
            alerts.append("📐 ÁNGULO CRÍTICO: Trayectoria fuera de límites")
            self.mission_status = "ALERTA"
        
        return alerts
    
    def reentry_equations_advanced(self, t, y):
        """Sistema avanzado de ecuaciones diferenciales mejorado"""
        v, gamma, h, x, bank_angle = y
        
        # Prevenir valores no físicos
        h = max(h, 0)
        v = max(v, 0.1)
        
        # Propiedades atmosféricas
        rho = self.atmospheric_density(h)
        g = self.gravity(h)
        mach = v / 340
        
        # Coeficientes aerodinámicos mejorados
        CD = self.drag_coefficient(mach, gamma)
        CL = self.lift_coefficient(mach, bank_angle)
        
        # Fuerzas aerodinámicas
        q_dynamic = 0.5 * rho * v**2
        drag = q_dynamic * CD * self.A
        lift = q_dynamic * CL * self.A
        
        # Control adaptativo mejorado
        bank_angle, control_signal, ml_control = self.adaptive_controller(
            [v, gamma, h, x, bank_angle], 
            [self.target_g_force, self.target_temp]
        )
        
        # Ecuaciones de movimiento con sustentación mejoradas
        dv_dt = -drag / self.m - g * np.sin(gamma)
        dgamma_dt = (lift * np.cos(bank_angle) / (self.m * v) + 
                     v * np.cos(gamma) / (self.R_earth + h) - 
                     g * np.cos(gamma) / v)
        dh_dt = -v * np.sin(gamma)
        dx_dt = v * np.cos(gamma) * (self.R_earth / (self.R_earth + h))
        dbank_dt = 0  # Controlado discretamente
        
        # Calcular parámetros térmicos mejorados
        heat_flux = self.heat_flux_calculation(rho, v)
        T_surface, T_internal, ablation_rate = self.thermal_protection_system(
            heat_flux, 0.1, getattr(self, 'current_temp', 300)
        )
        self.current_temp = T_surface
        
        # Verificar blackout
        blackout = self.check_blackout_conditions(h, v)
        if blackout:
            self.blackout_duration += 0.1
            self.communication_blackout = True
        
        # Verificar condiciones críticas
        g_force_current = drag / (self.m * self.g0)
        alerts = self.check_critical_conditions(v, gamma, h, T_surface, g_force_current, T_internal)
        self.mission_alerts.extend(alerts)
        
        # Actualizar métricas
        self.max_heat_flux = max(self.max_heat_flux, heat_flux)
        self.max_g_force = max(self.max_g_force, g_force_current)
        self.max_temp = max(self.max_temp, T_surface)
        self.max_internal_temp = max(self.max_internal_temp, T_internal)
        self.total_ablation += ablation_rate * 0.1
        
        return [dv_dt, dgamma_dt, dh_dt, dx_dt, dbank_dt]
    
    def train_ml_controller(self):
        """Entrenar controlador de Machine Learning mejorado"""
        print("🤖 Entrenando sistema de ML avanzado...")
        
        # Generar datos de entrenamiento más robustos
        training_inputs = []
        training_outputs = []
        
        for _ in range(500):  # Más datos de entrenamiento
            # Condiciones aleatorias más variadas
            v_rand = np.random.uniform(1000, 8000)
            h_rand = np.random.uniform(5000, 120000)
            gamma_rand = np.random.uniform(-25, -1)
            
            rho = self.atmospheric_density(h_rand)
            heat_flux = self.heat_flux_calculation(rho, v_rand)
            g_force = 0.5 * rho * v_rand**2 * self.CD_base * self.A / (self.m * self.g0)
            
            # Features normalizadas
            features = [
                v_rand/8000, 
                h_rand/120000, 
                gamma_rand/-25, 
                g_force/10, 
                heat_flux/1e6
            ]
            training_inputs.append(features)
            
            # Control óptimo mejorado (función más compleja)
            error_g = g_force - self.target_g_force
            error_temp = heat_flux/1e6 - self.target_temp/1000
            
            optimal_control = np.tanh(error_g * 0.2 + error_temp * 0.1)
            training_outputs.append(optimal_control)
        
        # Entrenar red neuronal mejorada
        X = np.array(training_inputs)
        y = np.array(training_outputs)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.ml_controller = MLPRegressor(
            hidden_layer_sizes=(50, 30, 20, 10),
            activation='tanh',
            solver='adam',
            alpha=0.001,
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        self.ml_controller.fit(X_scaled, y)
        
        # Evaluar performance
        train_score = self.ml_controller.score(X_scaled, y)
        print(f"✅ Entrenamiento ML completado - R² Score: {train_score:.3f}")
    
    def landing_event(self, t, y):
        """Evento de aterrizaje para detener la simulación"""
        return y[2]  # Altitud
    
    # Configurar propiedades del evento después de definirlo
    landing_event.terminal = True
    landing_event.direction = -1
    
    def simulate_mission(self, v0=7800, gamma0=-6, h0=120e3, x0=0, t_max=2000):
        """Simulación completa de misión mejorada"""
        print("🚀 INICIANDO MISIÓN SRAI AVANZADA v2.0")
        print("="*60)
        
        # Entrenar ML si no está entrenado
        if self.ml_controller is None:
            self.train_ml_controller()
        
        # Condiciones iniciales [v, gamma, h, x, bank_angle]
        y0 = [v0, np.radians(gamma0), h0, x0, 0]
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, 2000)
        
        # Resolver sistema de ecuaciones con evento de aterrizaje
        print("📊 Ejecutando simulación numérica avanzada...")
        sol = solve_ivp(self.reentry_equations_advanced, t_span, y0, 
                       t_eval=t_eval, rtol=1e-8, atol=1e-10,
                       max_step=1.0, events=[self.landing_event])
        
        # Procesar telemetría mejorada
        self.process_advanced_telemetry(sol)
        
        # Estadísticas finales mejoradas
        self.print_mission_summary()
        
        return sol
    
    def process_advanced_telemetry(self, sol):
        """Procesar telemetría avanzada con todas las métricas"""
        self.telemetry = {key: [] for key in self.telemetry.keys()}
        
        for i, t in enumerate(sol.t):
            v, gamma, h, x, bank = sol.y[:, i]
            
            # Básicos
            self.telemetry['time'].append(t)
            self.telemetry['velocity'].append(v)
            self.telemetry['gamma'].append(np.degrees(gamma))
            self.telemetry['altitude'].append(h)
            self.telemetry['distance'].append(x)
            self.telemetry['bank_angle'].append(np.degrees(bank))
            
            # Calculados avanzados
            rho = self.atmospheric_density(h)
            mach = v / 340
            q_dynamic = 0.5 * rho * v**2
            heat_flux = self.heat_flux_calculation(rho, v)
            g_force = 0.5 * rho * v**2 * self.drag_coefficient(mach, gamma) * self.A / (self.m * self.g0)
            T_surf, T_int, abl_rate = self.thermal_protection_system(heat_flux, 1.0)
            
            # Verificaciones
            blackout = self.check_blackout_conditions(h, v)
            comm_status = "NOMINAL" if not blackout else "BLACKOUT"
            structural_load = g_force * self.m / 1000  # kN
            
            # Agregar a telemetría
            self.telemetry['heat_flux'].append(heat_flux)
            self.telemetry['g_force'].append(g_force)
            self.telemetry['temperature'].append(T_surf)
            self.telemetry['internal_temp'].append(T_int)
            self.telemetry['thermal_stress'].append(T_surf - T_int)
            self.telemetry['ablation_rate'].append(abl_rate)
            self.telemetry['drag_coefficient'].append(self.drag_coefficient(mach, gamma))
            self.telemetry['blackout'].append(blackout)
            self.telemetry['mach_number'].append(mach)
            self.telemetry['dynamic_pressure'].append(q_dynamic)
            self.telemetry['structural_load'].append(structural_load)
            self.telemetry['communication_status'].append(comm_status)
            
            # Control
            _, control_sig, ml_adj = self.adaptive_controller([v, gamma, h, x, bank], 
                                                           [self.target_g_force, self.target_temp])
            self.telemetry['control_signal'].append(control_sig)
            self.telemetry['ml_bank_adjustment'].append(ml_adj)
    
    def print_mission_summary(self):
        """Imprimir resumen detallado de la misión"""
        print(f"\n📈 RESUMEN COMPLETO DE MISIÓN SRAI:")
        print("="*60)
        print(f"🕒 Estado final: {self.mission_status}")
        print(f"🛬 Altitud final: {self.telemetry['altitude'][-1]/1000:.2f} km")
        print(f"🚀 Velocidad final: {self.telemetry['velocity'][-1]:.1f} m/s")
        print(f"📍 Distancia recorrida: {self.telemetry['distance'][-1]/1000:.1f} km")
        print(f"⏱️  Duración total: {self.telemetry['time'][-1]:.1f} s")
        
        print(f"\n🌡️  MÉTRICAS TÉRMICAS:")
        print(f"   • Temperatura máxima: {self.max_temp:.1f} K")
        print(f"   • Temp. interna máxima: {self.max_internal_temp:.1f} K")
        print(f"   • Flujo de calor máximo: {self.max_heat_flux/1e6:.2f} MW/m²")
        print(f"   • Ablación total: {self.total_ablation*1000:.2f} mm")
        print(f"   • Integridad térmica: {'✅ OK' if not self.thermal_failure else '❌ FALLO'}")
        
        print(f"\n🏃 MÉTRICAS DINÁMICAS:")
        print(f"   • Fuerza G máxima: {self.max_g_force:.2f} G")
        print(f"   • Mach máximo: {max(self.telemetry['mach_number']):.1f}")
        print(f"   • Presión dinámica máx: {max(self.telemetry['dynamic_pressure'])/1000:.1f} kPa")
        print(f"   • Integridad estructural: {'✅ OK' if not self.structural_failure else '❌ FALLO'}")
        
        print(f"\n📡 COMUNICACIONES:")
        print(f"   • Duración blackout: {self.blackout_duration:.1f} s")
        print(f"   • Estado comunicación: {'✅ NOMINAL' if not self.communication_blackout else '📵 PERDIDA'}")
        
        print(f"\n🤖 SISTEMA DE CONTROL:")
        print(f"   • Control ML: {'✅ ACTIVO' if self.ml_controller else '❌ INACTIVO'}")
        print(f"   • Ruido sensores: {'✅ HABILITADO' if self.sensor_noise else '❌ DESHABILITADO'}")
        print(f"   • Alertas generadas: {len(set(self.mission_alerts))}")
        
        if self.mission_alerts:
            print(f"\n⚠️  ALERTAS DE MISIÓN:")
            for alert in set(self.mission_alerts):
                print(f"   • {alert}")
    
    def create_advanced_dashboard(self):
        """Dashboard avanzado con todas las métricas mejorado"""
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('🚀 SISTEMA DE REENTRADA ATMOSFÉRICA INTELIGENTE v2.0 - DASHBOARD COMPLETO', 
                     fontsize=20, fontweight='bold', color='white')
        fig.patch.set_facecolor('black')
        
        # Grid de subplots expandido
        gs = fig.add_gridspec(5, 8, hspace=0.4, wspace=0.4)
        
        # 1. Trayectoria 3D mejorada
        ax1 = fig.add_subplot(gs[0:2, 0:4])
        ax1.plot(np.array(self.telemetry['distance'])/1000, 
                np.array(self.telemetry['altitude'])/1000, 
                'cyan', linewidth=3, label='Trayectoria Real')
        
        # Marcar fases de la misión
        blackout_indices = [i for i, bo in enumerate(self.telemetry['blackout']) if bo]
        if blackout_indices:
            ax1.scatter([self.telemetry['distance'][i]/1000 for i in blackout_indices[:10]], 
                       [self.telemetry['altitude'][i]/1000 for i in blackout_indices[:10]], 
                       c='red', s=20, label='Blackout Zone', alpha=0.7)
        
        ax1.fill_between(np.array(self.telemetry['distance'])/1000, 
                        np.array(self.telemetry['altitude'])/1000,
                        alpha=0.2, color='cyan')
        ax1.set_xlabel('Distancia (km)', color='white', fontsize=12)
        ax1.set_ylabel('Altitud (km)', color='white', fontsize=12)
        ax1.set_title('TRAYECTORIA DE REENTRADA CON FASES', color='lime', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('black')
        ax1.tick_params(colors='white')
        
        # 2-5. Métricas principales
        metrics = [
            ('velocity', 'VELOCIDAD', 'red', 'km/s', 1000),
            ('temperature', 'TEMPERATURA', 'orange', 'K', 1),
            ('g_force', 'ACELERACIÓN', 'magenta', 'G', 1),
            ('mach_number', 'NÚMERO MACH', 'yellow', 'Mach', 1)
        ]
        
        for i, (key, title, color, unit, divisor) in enumerate(metrics):
            ax = fig.add_subplot(gs[0, 4+i])
            ax.plot(self.telemetry['time'], np.array(self.telemetry[key])/divisor, 
                    color, linewidth=2)
            if key == 'g_force':
                ax.axhline(y=self.target_g_force, color='yellow', linestyle='--', alpha=0.7)
    def landing_event(self, t, y):
        return y[2]  # Altitud
    
    # Configurar propiedades del evento inmediatamente después de la definición
    landing_event.terminal = True
    landing_event.direction = -1
    
    def simulate_mission(self, v0=7800, gamma0=-6, h0=120e3, x0=0, t_max=2000):
        """Simulación completa de misión mejorada"""
        print("🚀 INICIANDO MISIÓN SRAI AVANZADA v2.0")
        print("="*60)
        
        # Entrenar ML si no está entrenado
        if self.ml_controller is None:
            self.train_ml_controller()
        
        # Condiciones iniciales [v, gamma, h, x, bank_angle]
        y0 = [v0, np.radians(gamma0), h0, x0, 0]
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, 2000)
        
        # Resolver sistema de ecuaciones con evento de aterrizaje
        print("📊 Ejecutando simulación numérica avanzada...")
        sol = solve_ivp(self.reentry_equations_advanced, t_span, y0, 
                       t_eval=t_eval, rtol=1e-8, atol=1e-10,
                       max_step=1.0, events=[self.landing_event])
        
        # Procesar telemetría mejorada
        self.process_advanced_telemetry(sol)
        
        # Estadísticas finales mejoradas
        self.print_mission_summary()
        
        return sol
    
    def process_advanced_telemetry(self, sol):
        """Procesar telemetría avanzada con todas las métricas"""
        self.telemetry = {key: [] for key in self.telemetry.keys()}
        
        for i, t in enumerate(sol.t):
            v, gamma, h, x, bank = sol.y[:, i]
            
            # Básicos
            self.telemetry['time'].append(t)
            self.telemetry['velocity'].append(v)
            self.telemetry['gamma'].append(np.degrees(gamma))
            self.telemetry['altitude'].append(h)
            self.telemetry['distance'].append(x)
            self.telemetry['bank_angle'].append(np.degrees(bank))
            
            # Calculados avanzados
            rho = self.atmospheric_density(h)
            mach = v / 340
            q_dynamic = 0.5 * rho * v**2
            heat_flux = self.heat_flux_calculation(rho, v)
            g_force = 0.5 * rho * v**2 * self.drag_coefficient(mach, gamma) * self.A / (self.m * self.g0)
            T_surf, T_int, abl_rate = self.thermal_protection_system(heat_flux, 1.0)
            
            # Verificaciones
            blackout = self.check_blackout_conditions(h, v)
            comm_status = "NOMINAL" if not blackout else "BLACKOUT"
            structural_load = g_force * self.m / 1000  # kN
            
            # Agregar a telemetría
            self.telemetry['heat_flux'].append(heat_flux)
            self.telemetry['g_force'].append(g_force)
            self.telemetry['temperature'].append(T_surf)
            self.telemetry['internal_temp'].append(T_int)
            self.telemetry['thermal_stress'].append(T_surf - T_int)
            self.telemetry['ablation_rate'].append(abl_rate)
            self.telemetry['drag_coefficient'].append(self.drag_coefficient(mach, gamma))
            self.telemetry['blackout'].append(blackout)
            self.telemetry['mach_number'].append(mach)
            self.telemetry['dynamic_pressure'].append(q_dynamic)
            self.telemetry['structural_load'].append(structural_load)
            self.telemetry['communication_status'].append(comm_status)
            
            # Control
            _, control_sig, ml_adj = self.adaptive_controller([v, gamma, h, x, bank], 
                                                           [self.target_g_force, self.target_temp])
            self.telemetry['control_signal'].append(control_sig)
            self.telemetry['ml_bank_adjustment'].append(ml_adj)
    
    def print_mission_summary(self):
        """Imprimir resumen detallado de la misión"""
        print(f"\n📈 RESUMEN COMPLETO DE MISIÓN SRAI:")
        print("="*60)
        print(f"🕒 Estado final: {self.mission_status}")
        print(f"🛬 Altitud final: {self.telemetry['altitude'][-1]/1000:.2f} km")
        print(f"🚀 Velocidad final: {self.telemetry['velocity'][-1]:.1f} m/s")
        print(f"📍 Distancia recorrida: {self.telemetry['distance'][-1]/1000:.1f} km")
        print(f"⏱️  Duración total: {self.telemetry['time'][-1]:.1f} s")
        
        print(f"\n🌡️  MÉTRICAS TÉRMICAS:")
        print(f"   • Temperatura máxima: {self.max_temp:.1f} K")
        print(f"   • Temp. interna máxima: {self.max_internal_temp:.1f} K")
        print(f"   • Flujo de calor máximo: {self.max_heat_flux/1e6:.2f} MW/m²")
        print(f"   • Ablación total: {self.total_ablation*1000:.2f} mm")
        print(f"   • Integridad térmica: {'✅ OK' if not self.thermal_failure else '❌ FALLO'}")
        
        print(f"\n🏃 MÉTRICAS DINÁMICAS:")
        print(f"   • Fuerza G máxima: {self.max_g_force:.2f} G")
        print(f"   • Mach máximo: {max(self.telemetry['mach_number']):.1f}")
        print(f"   • Presión dinámica máx: {max(self.telemetry['dynamic_pressure'])/1000:.1f} kPa")
        print(f"   • Integridad estructural: {'✅ OK' if not self.structural_failure else '❌ FALLO'}")
        
        print(f"\n📡 COMUNICACIONES:")
        print(f"   • Duración blackout: {self.blackout_duration:.1f} s")
        print(f"   • Estado comunicación: {'✅ NOMINAL' if not self.communication_blackout else '📵 PERDIDA'}")
        
        print(f"\n🤖 SISTEMA DE CONTROL:")
        print(f"   • Control ML: {'✅ ACTIVO' if self.ml_controller else '❌ INACTIVO'}")
        print(f"   • Ruido sensores: {'✅ HABILITADO' if self.sensor_noise else '❌ DESHABILITADO'}")
        print(f"   • Alertas generadas: {len(set(self.mission_alerts))}")
        
        if self.mission_alerts:
            print(f"\n⚠️  ALERTAS DE MISIÓN:")
            for alert in set(self.mission_alerts):
                print(f"   • {alert}")
    
    def create_advanced_dashboard(self):
        """Dashboard avanzado con todas las métricas mejorado"""
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('🚀 SISTEMA DE REENTRADA ATMOSFÉRICA INTELIGENTE v2.0 - DASHBOARD COMPLETO', 
                     fontsize=20, fontweight='bold', color='white')
        fig.patch.set_facecolor('black')
        
        # Grid de subplots expandido
        gs = fig.add_gridspec(5, 8, hspace=0.4, wspace=0.4)
        
        # 1. Trayectoria 3D mejorada
        ax1 = fig.add_subplot(gs[0:2, 0:4])
        ax1.plot(np.array(self.telemetry['distance'])/1000, 
                np.array(self.telemetry['altitude'])/1000, 
                'cyan', linewidth=3, label='Trayectoria Real')
        
        # Marcar fases de la misión
        blackout_indices = [i for i, bo in enumerate(self.telemetry['blackout']) if bo]
        if blackout_indices:
            ax1.scatter([self.telemetry['distance'][i]/1000 for i in blackout_indices[:10]], 
                       [self.telemetry['altitude'][i]/1000 for i in blackout_indices[:10]], 
                       c='red', s=20, label='Blackout Zone', alpha=0.7)
        
        ax1.fill_between(np.array(self.telemetry['distance'])/1000, 
                        np.array(self.telemetry['altitude'])/1000,
                        alpha=0.2, color='cyan')
        ax1.set_xlabel('Distancia (km)', color='white', fontsize=12)
        ax1.set_ylabel('Altitud (km)', color='white', fontsize=12)
        ax1.set_title('TRAYECTORIA DE REENTRADA CON FASES', color='lime', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('black')
        ax1.tick_params(colors='white')
        
        # 2-5. Métricas principales
        metrics = [
            ('velocity', 'VELOCIDAD', 'red', 'km/s', 1000),
            ('temperature', 'TEMPERATURA', 'orange', 'K', 1),
            ('g_force', 'ACELERACIÓN', 'magenta', 'G', 1),
            ('mach_number', 'NÚMERO MACH', 'yellow', 'Mach', 1)
        ]
        
        for i, (key, title, color, unit, divisor) in enumerate(metrics):
            ax = fig.add_subplot(gs[0, 4+i])
            ax.plot(self.telemetry['time'], np.array(self.telemetry[key])/divisor, 
                    color, linewidth=2)
            if key == 'g_force':
                ax.axhline(y=self.target_g_force, color='yellow', linestyle='--', alpha=0.7)
            ax.set_title(f'{title}\nMáx: {max(self.telemetry[key])/divisor:.1f} {unit}', 
                        color=color, fontweight='bold', fontsize=10)
            ax.set_facecolor('black')
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(True, alpha=0.3)
        
        # 6. Flujo de calor vs tiempo
        ax6 = fig.add_subplot(gs[1, 4:6])
        heat_flux_MW = np.array(self.telemetry['heat_flux'])/1e6
        ax6.plot(self.telemetry['time'], heat_flux_MW, 'orange', linewidth=2)
        ax6.fill_between(self.telemetry['time'], heat_flux_MW, alpha=0.3, color='orange')
        ax6.set_title('FLUJO DE CALOR\nMáx: {:.2f} MW/m²'.format(max(heat_flux_MW)), 
                     color='orange', fontweight='bold', fontsize=10)
        ax6.set_ylabel('MW/m²', color='white', fontsize=8)
        ax6.set_facecolor('black')
        ax6.tick_params(colors='white', labelsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 7. Temperaturas comparativas
        ax7 = fig.add_subplot(gs[1, 6:8])
        ax7.plot(self.telemetry['time'], self.telemetry['temperature'], 'red', 
                linewidth=2, label='Superficie')
        ax7.plot(self.telemetry['time'], self.telemetry['internal_temp'], 'blue', 
                linewidth=2, label='Interna')
        ax7.axhline(y=self.max_internal_temp, color='yellow', linestyle='--', alpha=0.7)
        ax7.set_title('TEMPERATURAS\nSup/Int Máx: {:.0f}/{:.0f} K'.format(
            max(self.telemetry['temperature']), max(self.telemetry['internal_temp'])), 
                     color='red', fontweight='bold', fontsize=10)
        ax7.legend(fontsize=8)
        ax7.set_facecolor('black')
        ax7.tick_params(colors='white', labelsize=8)
        ax7.grid(True, alpha=0.3)
        
        # 8. Control ML y señales
        ax8 = fig.add_subplot(gs[2, 0:2])
        ax8.plot(self.telemetry['time'], self.telemetry['control_signal'], 'green', 
                linewidth=2, label='Señal Control')
        ax8.plot(self.telemetry['time'], np.array(self.telemetry['ml_bank_adjustment'])*10, 
                'purple', linewidth=2, label='ML Ajuste x10')
        ax8.set_title('SISTEMA DE CONTROL INTELIGENTE', color='green', fontweight='bold', fontsize=12)
        ax8.legend()
        ax8.set_xlabel('Tiempo (s)', color='white')
        ax8.set_facecolor('black')
        ax8.tick_params(colors='white')
        ax8.grid(True, alpha=0.3)
        
        # 9. Ángulo de banco
        ax9 = fig.add_subplot(gs[2, 2:4])
        ax9.plot(self.telemetry['time'], self.telemetry['bank_angle'], 'cyan', linewidth=2)
        ax9.set_title('ÁNGULO DE BANCO', color='cyan', fontweight='bold', fontsize=12)
        ax9.set_ylabel('Grados', color='white')
        ax9.set_xlabel('Tiempo (s)', color='white')
        ax9.set_facecolor('black')
        ax9.tick_params(colors='white')
        ax9.grid(True, alpha=0.3)
        
        # 10. Coeficiente de arrastre dinámico
        ax10 = fig.add_subplot(gs[2, 4:6])
        ax10.plot(self.telemetry['time'], self.telemetry['drag_coefficient'], 'magenta', linewidth=2)
        ax10.set_title('COEFICIENTE DE ARRASTRE', color='magenta', fontweight='bold', fontsize=12)
        ax10.set_ylabel('CD', color='white')
        ax10.set_xlabel('Tiempo (s)', color='white')
        ax10.set_facecolor('black')
        ax10.tick_params(colors='white')
        ax10.grid(True, alpha=0.3)
        
        # 11. Presión dinámica
        ax11 = fig.add_subplot(gs[2, 6:8])
        q_dynamic_kPa = np.array(self.telemetry['dynamic_pressure'])/1000
        ax11.plot(self.telemetry['time'], q_dynamic_kPa, 'yellow', linewidth=2)
        ax11.fill_between(self.telemetry['time'], q_dynamic_kPa, alpha=0.3, color='yellow')
        ax11.set_title('PRESIÓN DINÁMICA\nMáx: {:.1f} kPa'.format(max(q_dynamic_kPa)), 
                      color='yellow', fontweight='bold', fontsize=12)
        ax11.set_ylabel('kPa', color='white')
        ax11.set_xlabel('Tiempo (s)', color='white')
        ax11.set_facecolor('black')
        ax11.tick_params(colors='white')
        ax11.grid(True, alpha=0.3)
        
        # 12. Estado de comunicaciones
        ax12 = fig.add_subplot(gs[3, 0:2])
        blackout_binary = [1 if bo else 0 for bo in self.telemetry['blackout']]
        ax12.fill_between(self.telemetry['time'], blackout_binary, alpha=0.7, color='red')
        ax12.set_title('BLACKOUT DE COMUNICACIÓN', color='red', fontweight='bold', fontsize=12)
        ax12.set_ylabel('Estado', color='white')
        ax12.set_xlabel('Tiempo (s)', color='white')
        ax12.set_ylim(-0.1, 1.1)
        ax12.set_facecolor('black')
        ax12.tick_params(colors='white')
        ax12.grid(True, alpha=0.3)
        
        # 13. Carga estructural
        ax13 = fig.add_subplot(gs[3, 2:4])
        ax13.plot(self.telemetry['time'], self.telemetry['structural_load'], 'orange', linewidth=2)
        ax13.set_title('CARGA ESTRUCTURAL', color='orange', fontweight='bold', fontsize=12)
        ax13.set_ylabel('kN', color='white')
        ax13.set_xlabel('Tiempo (s)', color='white')
        ax13.set_facecolor('black')
        ax13.tick_params(colors='white')
        ax13.grid(True, alpha=0.3)
        
        # 14. Tasa de ablación
        ax14 = fig.add_subplot(gs[3, 4:6])
        ablation_mm = np.array(self.telemetry['ablation_rate'])*1000
        ax14.plot(self.telemetry['time'], ablation_mm, 'red', linewidth=2)
        ax14.fill_between(self.telemetry['time'], ablation_mm, alpha=0.3, color='red')
        ax14.set_title('TASA DE ABLACIÓN\nTotal: {:.2f} mm'.format(self.total_ablation*1000), 
                      color='red', fontweight='bold', fontsize=12)
        ax14.set_ylabel('mm/s', color='white')
        ax14.set_xlabel('Tiempo (s)', color='white')
        ax14.set_facecolor('black')
        ax14.tick_params(colors='white')
        ax14.grid(True, alpha=0.3)
        
        # 15. Estrés térmico
        ax15 = fig.add_subplot(gs[3, 6:8])
        ax15.plot(self.telemetry['time'], self.telemetry['thermal_stress'], 'purple', linewidth=2)
        ax15.set_title('ESTRÉS TÉRMICO', color='purple', fontweight='bold', fontsize=12)
        ax15.set_ylabel('ΔT (K)', color='white')
        ax15.set_xlabel('Tiempo (s)', color='white')
        ax15.set_facecolor('black')
        ax15.tick_params(colors='white')
        ax15.grid(True, alpha=0.3)
        
        # 16. Panel de estado de misión
        ax16 = fig.add_subplot(gs[4, 0:8])
        ax16.text(0.1, 0.8, f'🕒 ESTADO: {self.mission_status}', color='lime', fontsize=16, fontweight='bold')
        ax16.text(0.1, 0.6, f'🛬 Altitud final: {self.telemetry["altitude"][-1]/1000:.2f} km', color='white', fontsize=12)
        ax16.text(0.1, 0.4, f'🚀 Velocidad final: {self.telemetry["velocity"][-1]:.1f} m/s', color='white', fontsize=12)
        ax16.text(0.1, 0.2, f'⏱️ Duración: {self.telemetry["time"][-1]:.1f} s', color='white', fontsize=12)
        
        ax16.text(0.4, 0.8, f'🌡️  Temp. máx: {self.max_temp:.0f} K', color='orange', fontsize=12)
        ax16.text(0.4, 0.6, f'🏃 G máx: {self.max_g_force:.2f} G', color='magenta', fontsize=12)
        ax16.text(0.4, 0.4, f'🔥 Flujo máx: {self.max_heat_flux/1e6:.2f} MW/m²', color='red', fontsize=12)
        ax16.text(0.4, 0.2, f'📡 Blackout: {self.blackout_duration:.1f} s', color='yellow', fontsize=12)
        
        # Indicadores de estado
        thermal_status = "✅ OK" if not self.thermal_failure else "❌ FALLO"
        structural_status = "✅ OK" if not self.structural_failure else "❌ FALLO"
        
        ax16.text(0.7, 0.8, f'🛡️ Térmica: {thermal_status}', color='lime' if not self.thermal_failure else 'red', fontsize=12)
        ax16.text(0.7, 0.6, f'🏗️ Estructural: {structural_status}', color='lime' if not self.structural_failure else 'red', fontsize=12)
        ax16.text(0.7, 0.4, f'🤖 ML Activo: {"✅" if self.ml_controller else "❌"}', color='cyan', fontsize=12)
        ax16.text(0.7, 0.2, f'📊 Alertas: {len(set(self.mission_alerts))}', color='yellow', fontsize=12)
        
        ax16.set_xlim(0, 1)
        ax16.set_ylim(0, 1)
        ax16.set_facecolor('black')
        ax16.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_animated_trajectory(self):
        """Crear animación de la trayectoria"""
        print("🎬 Creando animación de trayectoria...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('black')
        
        # Configurar subplots
        ax1.set_facecolor('black')
        ax2.set_facecolor('black')
        
        # Trayectoria completa (referencia)
        ax1.plot(np.array(self.telemetry['distance'])/1000, 
                np.array(self.telemetry['altitude'])/1000, 
                'gray', alpha=0.5, linewidth=1)
        
        # Líneas animadas
        line1, = ax1.plot([], [], 'cyan', linewidth=3)
        point1, = ax1.plot([], [], 'ro', markersize=8)
        
        # Gráfico de velocidad
        ax2.plot(self.telemetry['time'], self.telemetry['velocity'], 'gray', alpha=0.5)
        line2, = ax2.plot([], [], 'red', linewidth=3)
        point2, = ax2.plot([], [], 'ro', markersize=8)
        
        # Configurar ejes
        ax1.set_xlabel('Distancia (km)', color='white')
        ax1.set_ylabel('Altitud (km)', color='white')
        ax1.set_title('TRAYECTORIA ANIMADA', color='cyan', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(colors='white')
        
        ax2.set_xlabel('Tiempo (s)', color='white')
        ax2.set_ylabel('Velocidad (m/s)', color='white')
        ax2.set_title('PERFIL DE VELOCIDAD', color='red', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(colors='white')
        
        def animate(frame):
            # Actualizar trayectoria
            x_data = np.array(self.telemetry['distance'][:frame])/1000
            y_data = np.array(self.telemetry['altitude'][:frame])/1000
            line1.set_data(x_data, y_data)
            
            if frame > 0:
                point1.set_data([x_data[-1]], [y_data[-1]])
            
            # Actualizar velocidad
            t_data = self.telemetry['time'][:frame]
            v_data = self.telemetry['velocity'][:frame]
            line2.set_data(t_data, v_data)
            
            if frame > 0:
                point2.set_data([t_data[-1]], [v_data[-1]])
            
            return line1, point1, line2, point2
        
        # Crear animación
        anim = FuncAnimation(fig, animate, frames=len(self.telemetry['time']), 
                           interval=20, blit=True, repeat=True)
        
        return anim
    
    def export_telemetry_report(self):
        """Exportar reporte completo de telemetría"""
        print("📊 Generando reporte de telemetría...")
        
        # Crear DataFrame
        df = pd.DataFrame(self.telemetry)
        
        # Agregar métricas calculadas
        df['altitude_km'] = df['altitude'] / 1000
        df['velocity_kms'] = df['velocity'] / 1000
        df['heat_flux_MW'] = df['heat_flux'] / 1e6
        df['dynamic_pressure_kPa'] = df['dynamic_pressure'] / 1000
        
        # Estadísticas del reporte
        report = {
            'mission_summary': {
                'status': self.mission_status,
                'duration': self.telemetry['time'][-1],
                'final_altitude': self.telemetry['altitude'][-1],
                'final_velocity': self.telemetry['velocity'][-1],
                'distance_traveled': self.telemetry['distance'][-1],
                'blackout_duration': self.blackout_duration
            },
            'thermal_metrics': {
                'max_temperature': self.max_temp,
                'max_internal_temp': self.max_internal_temp,
                'max_heat_flux': self.max_heat_flux,
                'total_ablation': self.total_ablation,
                'thermal_failure': self.thermal_failure
            },
            'dynamic_metrics': {
                'max_g_force': self.max_g_force,
                'max_mach': max(self.telemetry['mach_number']),
                'max_dynamic_pressure': max(self.telemetry['dynamic_pressure']),
                'structural_failure': self.structural_failure
            },
            'control_metrics': {
                'ml_controller_active': self.ml_controller is not None,
                'sensor_noise_enabled': self.sensor_noise,
                'alerts_generated': len(set(self.mission_alerts))
            }
        }
        
        # Guardar datos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'telemetry_srai_{timestamp}.csv', index=False)
        
        with open(f'mission_report_srai_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"✅ Reporte guardado: telemetry_srai_{timestamp}.csv")
        print(f"✅ Resumen guardado: mission_report_srai_{timestamp}.json")
        
        return df, report
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo del sistema"""
        print("🚀 INICIANDO ANÁLISIS COMPLETO SRAI v2.0")
        print("="*60)
        
        try:
            # 1. Ejecutar simulación principal
            sol = self.simulate_mission()
            
            # 2. Crear dashboard
            print("📊 Generando dashboard avanzado...")
            dashboard = self.create_advanced_dashboard()
            
            # 3. Exportar datos
            df, report = self.export_telemetry_report()
            
            print("\n✅ ANÁLISIS COMPLETO FINALIZADO")
            print("="*60)
            
            return sol, df
            
        except Exception as e:
            print(f"❌ ERROR EN ANÁLISIS: {str(e)}")
            return None, None

# Configuración global de matplotlib para mejor rendimiento
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

def main():
    """Función principal del sistema SRAI"""
    try:
        print("🔧 Inicializando sistema avanzado...")
        srai = AdvancedAtmosphericReentrySystem()
        
        # Ejecutar análisis completo
        sol, telemetry_df = srai.run_complete_analysis()
        
        if sol is not None:
            print("\n🎯 MISIÓN COMPLETADA EXITOSAMENTE")
            print("📈 Mostrando visualizaciones...")
            plt.show()
        else:
            print("❌ Error en la simulación")
            
    except Exception as e:
        print(f"❌ Error en la ejecución del sistema: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    sistema, solucion, telemetria = main()
