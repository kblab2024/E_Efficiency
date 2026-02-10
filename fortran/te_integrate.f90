! te_integrate.f90
! k_parallel integration for TE Green's functions

module te_integrate
    use te_greens
    use bessel_functions
    use split_integration
    implicit none
    
contains

    ! Integrate G_yy over k_parallel to get real-space Green's function
    function gyy_TE_rho(n_list, d_list, N, layer_src, z_src, layer_obs, z_obs, &
                        k0, rho, k_parallel_max, n_gauss, safety_factor) result(G_rho)
        integer, intent(in) :: N, layer_src, layer_obs, n_gauss
        real(dp), intent(in) :: n_list(N), d_list(N-1), z_src, z_obs, k0, rho
        real(dp), intent(in) :: k_parallel_max, safety_factor
        complex(dp) :: G_rho
        
        real(dp) :: poles(N), intervals(2, N+1), kp_all(N * n_gauss * 2), w_all(N * n_gauss * 2)
        integer :: n_poles, n_intervals, n_total, i
        complex(dp) :: Gyy_kp(N * n_gauss * 2), J0_vals(N * n_gauss * 2)
        complex(dp) :: integrand(N * n_gauss * 2), integral
        real(dp), parameter :: pi = 3.141592653589793238462643383279502884197_dp
        
        ! Find pole positions
        call find_poles(n_list, N, k0, poles, n_poles)
        
        ! Create integration intervals avoiding poles
        call create_intervals(0.0_dp, k_parallel_max, poles, n_poles, safety_factor, &
                             intervals, n_intervals)
        
        ! Get Gauss quadrature points and weights for all intervals
        call integrate_split_gauss(intervals, n_intervals, n_gauss, &
                                  kp_all, w_all, n_total)
        
        ! Evaluate G_yy at all quadrature points
        do i = 1, n_total
            Gyy_kp(i) = gyy_TE(n_list, d_list, N, layer_src, z_src, &
                               layer_obs, z_obs, k0, kp_all(i))
        end do
        
        ! Evaluate Bessel J0 at all points
        do i = 1, n_total
            J0_vals(i) = J0_series(cmplx(kp_all(i) * rho, 0.0_dp, kind=dp))
        end do
        
        ! Form integrand: (1/π) * kp * J0(kp*rho) * G_yy(kp)
        do i = 1, n_total
            integrand(i) = (kp_all(i) / pi) * J0_vals(i) * Gyy_kp(i)
        end do
        
        ! Perform integration (weighted sum)
        integral = cmplx(0.0_dp, 0.0_dp, kind=dp)
        do i = 1, n_total
            integral = integral + w_all(i) * integrand(i)
        end do
        
        G_rho = integral
    end function gyy_TE_rho

    ! Compute G_yy(rho) for an array of rho values (OpenMP parallel)
    subroutine gyy_TE_rho_array(n_list, d_list, N, layer_src, z_src, layer_obs, z_obs, &
                                k0, rho_arr, n_rho, k_parallel_max, n_gauss, safety_factor, &
                                G_rho_arr)
        integer, intent(in) :: N, layer_src, layer_obs, n_gauss, n_rho
        real(dp), intent(in) :: n_list(N), d_list(N-1), z_src, z_obs, k0
        real(dp), intent(in) :: rho_arr(n_rho), k_parallel_max, safety_factor
        complex(dp), intent(out) :: G_rho_arr(n_rho)
        
        integer :: i
        
        !$omp parallel do default(shared) private(i) schedule(dynamic)
        do i = 1, n_rho
            G_rho_arr(i) = gyy_TE_rho(n_list, d_list, N, layer_src, z_src, &
                                      layer_obs, z_obs, k0, rho_arr(i), &
                                      k_parallel_max, n_gauss, safety_factor)
        end do
        !$omp end parallel do
    end subroutine gyy_TE_rho_array

end module te_integrate

! Main program for TE integration demo
program main_te_integrate
    use te_integrate
    !$ use omp_lib
    implicit none
    
    integer, parameter :: N = 4, n_rho = 100, n_gauss = 32
    real(dp) :: n_list(N), d_list(N-1)
    real(dp) :: z_src, z_obs, k0, wl, k_parallel_max, safety_factor
    real(dp) :: rho_arr(n_rho), rho_min, rho_max
    complex(dp) :: G_rho_arr(n_rho)
    integer :: i
    real(dp), parameter :: pi = 3.141592653589793238462643383279502884197_dp
    
    ! Setup geometry
    n_list = [1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]
    d_list = [0.0_dp, 2.0_dp, 1.5_dp]
    
    ! Source and observation positions
    layer_src = 1
    layer_obs = 1
    z_src = -0.35_dp
    z_obs = -0.20_dp
    
    ! Wavelength and k0
    wl = 0.650_dp
    k0 = 2.0_dp * pi / wl
    
    ! Integration parameters
    k_parallel_max = 3.5_dp * k0
    safety_factor = 0.05_dp  ! 5% safety margin around poles
    
    ! rho array
    rho_min = 0.0_dp
    rho_max = 1.30_dp
    do i = 1, n_rho
        rho_arr(i) = rho_min + (rho_max - rho_min) * real(i-1, dp) / real(n_rho-1, dp)
    end do
    
    ! Compute G_yy(rho)
    print *, 'Computing G_yy(rho) using split Gauss quadrature...'
    print *, 'Number of Gauss points per interval:', n_gauss
    print *, 'Safety factor:', safety_factor
    !$ print *, 'OpenMP threads:', omp_get_max_threads()
    print *, ''
    
    call gyy_TE_rho_array(n_list, d_list, N, layer_src, z_src, layer_obs, z_obs, &
                          k0, rho_arr, n_rho, k_parallel_max, n_gauss, safety_factor, &
                          G_rho_arr)
    
    ! Write results to file
    open(unit=20, file='gyy_vs_rho.dat', status='replace')
    write(20, '(A)') '# rho  Re(Gyy)  Im(Gyy)  |Gyy|'
    do i = 1, n_rho
        write(20, '(4E16.8)') rho_arr(i), real(G_rho_arr(i)), aimag(G_rho_arr(i)), abs(G_rho_arr(i))
    end do
    close(20)
    
    print *, 'Results written to gyy_vs_rho.dat'
    print *, 'First few values:'
    do i = 1, min(5, n_rho)
        print '(A,F8.4,A,2E14.6)', '  rho=', rho_arr(i), '  G_yy=', G_rho_arr(i)
    end do
    
end program main_te_integrate
