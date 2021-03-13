classdef SimpleModel
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        inicio int32
        mes_inicio string
        fin int32
        mes_fin string
        num_datos int32
        tiempo string
        minimizer string
        t double
        t_pred double

        N {mustBeNumeric}
        sus {mustBeNumeric}
        inf {mustBeNumeric}
        rec {mustBeNumeric}
        exp {mustBeNumeric}

        data double
        sol double
        pred double
    end
    
    methods
        function obj = SimpleModel(tiempo,num_datos,mes_inicio,mes_fin)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.mes_inicio = mes_inicio;
            obj.mes_fin = mes_fin;
            obj.num_datos = num_datos;
            obj.tiempo = tiempo;

            inicio = 1;
            switch mes_inicio
                case "marzo"
                    inicio = 3;
                case "abril"
                    inicio = 34;
                case "mayo"
                    inicio = 64;
                case "junio"
                    inicio = 95;
                case "julio"
                    inicio = 125;
                case "agosto"
                    inicio = 156;
                case "septiembre"
                    inicio = 187;
                case "octubre"
                    inicio = 217;
                case "noviembre"
                    inicio = 248;
                case "diciembre"
                    inicio = 278;
                case "enero"
                    inicio = 309;
                otherwise
                    warning('Mes inicial inesperado')
            end
            obj.inicio = inicio;

            fin = 339;
            switch mes_fin
                case "marzo"
                    fin = 33;
                case "abril"
                    fin = 63;
                case "mayo"
                    fin = 94;
                case "junio"
                    fin = 124;
                case "julio"
                    fin = 155;
                case "agosto"
                    fin = 186;
                case "septiembre"
                    fin = 216;
                case "octubre"
                    fin = 247;
                case "noviembre"
                    fin = 277;
                case "diciembre"
                    fin = 308;
                case "enero"
                    fin = 339;
                otherwise
                    warning('Mes final inesperado')
            end
            obj.fin = fin;

            assert(inicio < fin, 'El mes final debe ser después del mes inicial')

            if num_datos ~= 0
                obj.t = linspace(0, num_datos, num_datos);
                %obj.t = [0, num_datos];
            else
                if tiempo == "semana"
                    semanas = round((fin - inicio + 1) / 7);
                    obj.t = linspace(0, semanas, semanas);
                    %obj.t = [0, semanas];
                elseif tiempo == "dia"
                    dia = fin - inicio + 1;
                    obj.t = linspace(0, dia, dia);
                    %obj.t = [0, dia];
                end
            end

        end
        
        function obj = official_data(obj,path)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            full_data = readmatrix(path);
            obj.N = full_data(obj.inicio, 8); % Población
            obj.sus = full_data(obj.inicio, 9); % Susceptibles
            obj.inf = full_data(obj.inicio, 5); % Infectados
            obj.rec = full_data(obj.inicio, 4); % Removidos
            obj.exp = full_data(obj.inicio, 7); % Expuestos

            data_holder = zeros(1, length(obj.t));
            indice = 1;

            if obj.num_datos ~= 0
                if obj.tiempo == "semana"
                    for semana = obj.inicio:7:obj.inicio+obj.num_datos
                        data_holder(indice) = full_data(semana, 5);
                        indice = indice + 1;
                    end
                elseif obj.tiempo == "dia"
                    for dia = obj.inicio:obj.inicio+obj.num_datos
                        data_holder(indice) = full_data(dia, 5);
                        indice = indice + 1;
                    end
                end
            else
                if obj.tiempo == "semana"
                    for semana = obj.inicio:7:obj.fin
                        data_holder(indice) = full_data(semana, 5);
                        indice = indice + 1;
                        % Esta condicion es agregada porque algunos meses tienen dias extras,
                        % que no cuentan como semanas, y por lo tanto no entren en los datos
                        if length(data_holder) + 1 > length(obj.t)
                            break;
                        end
                    end
                elseif obj.tiempo == "dia"
                    for dia = obj.inicio:obj.fin
                        data_holder(indice) = full_data(dia, 5);
                        indice = indice + 1;
                    end
                end
            end
            obj.data = data_holder;
        end

        function dydt = simple_model(~, t, y, N, alpha, beta, gamma, sigma)
            dydt = zeros(4,1);
            dydt(1) = (-alpha/N)*y(1)*(beta * y(4) + y(2));
            dydt(2) = gamma * y(4) - sigma * y(1);
            dydt(3) = sigma * y(2);
            dydt(4) = (alpha/N)*y(1)*(beta * y(4) + y(2)) - gamma * y(4);
        end

        function obj = solucion(obj, alpha, beta, gamma, sigma)
            [t,y] = ode45(...
                @(t,y) simple_model(obj, t, y, obj.N, alpha, beta, gamma, sigma),...
                obj.t,...
                [obj.sus obj.inf obj.rec obj.exp]...
            );
            obj.t = t;
            obj.sol = y;
        end

        % function obj = guardar_sol(obj, alpha, beta, gamma, sigma)
        %     [obj.t, obj.sol] = solucion(obj, alpha, beta, gamma);
        % end

        function dis =  distancia(obj,X)
            % Declara alpha y beta como variables y hace una predicción usando
            % el modelo.
            % Luego resta término a término el vector de predicciones y el vector
            % de datos reales, los eleva al cuadrado, suma el resultado y regresa
            % el resultado.
            alpha = X(1);
            beta = X(2);
            gamma = X(3);
            sigma = X(4);
            obj = solucion(obj, alpha, beta, gamma, sigma);
            pred = obj.sol(:,2);
            dis = 0;
            for index = 1:length(pred)
                dis = dis + (pred(index) - obj.data(index))^2;
            end
            dis = sqrt(dis);
            
        end

        function x = optimizar_con_limites(obj, x0, lb, up)
            % Optimiza los valores de alpha, beta, gamma y sigma
            % usando los límites ingresados en los vectores 'lb' y 'up'.
            % Valores de entrada:
            % x0: Vector con los puntos iniciales
            % lb: vector conteniendo los valores que actúan de límite inferior.
            % up: vector con los valores de límite superior. Debe tener la forma:
            %
            % Todos los vectores de entrada deben ser de la forma '''up = [Inf, 1, Inf, Inf]'''
            % donde la posición de cada valor representa a cada valor: el primer valor es el 
            % límite para alpha, el segundo para beta y los últimos dos para gamma y sigma.
            A = [];
            b = [];
            Aeq = [];
            beq = [];

            x = fmincon(@(X) distancia(obj,X), x0, A, b, Aeq, beq, lb, up);
        end

        function obj = predecir(obj, tiempo, cons)
            % Entrada:
            % tiempo: Int con el valor de días deseados a predecir.
            % cons: vector con los valores [N, alpha, beta, gamma, sigma]
            % los valores ingresados abajo corresponden a 
            % [susceptibles, infectados, removidos, expuestos]
            t_pred = linspace(0, tiempo, tiempo);
            [t,pred] = ode45(...
                @(t,y) simple_model(obj, t, y, cons(1), cons(2), cons(3), cons(4), cons(5)),...
                t_pred,...
                [102089743, 2041380, 1775427, 21705629]...
            );
            obj.t_pred = t;
            obj.pred = pred;
        end

        function graficar(obj, deseados, comparar, prediccion)
            t = 0;
            sol = 0;
            if prediccion
                t = obj.t_pred;
                sol = obj.pred;
            else
                t = obj.t;
                sol = obj.sol;
            end

            if ismember("s",deseados)
                plot(t, sol(:,1));
                legend('Susceptibles');
                hold on
            end
            if ismember("i", deseados)
                plot(t, sol(:,2));
                legend('Infectados');
                hold on
            end
            if ismember("r", deseados)
                plot(t, sol(:,3));
                legend('Recuperados');
                hold on
            end
            if ismember("e", deseados)
                plot(t, sol(:,4));
                legend('Expuestos');
                hold on
            end

            if prediccion
                legend('Location', 'best');
                xlabel('Tiempo (%s)',obj.tiempo);
                hold off
            else 
                if comparar
                    plot(t, obj.data, '-o');
                    legend('Infectados Reales');
                    hold on
                end
                legend('Location', 'best');
                xlabel('Tiempo');
                hold on
                if obj.mes_inicio == obj.mes_fin
                    title('Gráfica para el mes de ', obj.mes_inicio);
                    hold off
                else
                    title('Gráfica para los meses de  - %s', obj.mes_inicio, obj.mes_fin);
                    hold off
                end
            end
        end
    end
end

